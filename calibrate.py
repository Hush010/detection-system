"""Derive an honest, out-of-fold calibration for the detector's risk score.

Why this exists
---------------
`model.predict_proba` on a 252-sample TF-IDF + logistic-regression pipeline is
*not* a calibrated probability. Reading it as one produced the original bug:
`P(ai) + P(hybrid)` never drops far below 0.4 for genuine human writing, so
every human author scored ~50/100 and no submission could ever come back clean.

This script fits a one-dimensional Platt scaling on **out-of-fold** predictions
(never on data the fold's model saw) and derives the risk bands from the
false-positive rate observed on real human samples. The result is written to
`calibration.json`, which `detector.py` loads at runtime.

Everything here is plain JSON. No pickles are produced, so the runtime artifact
cannot execute code.

Run:  python calibrate.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from train_model import build_pipeline, load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
CALIBRATION_PATH = BASE_DIR / "calibration.json"

# A submission shorter than this cannot be judged. Industry detectors ask for
# comparable minimums; below it the model is reading noise.
MIN_WORDS = 50

# Share of genuine human submissions we are willing to see flagged. These are
# students, so the tolerance for a false accusation is deliberately low.
HIGH_RISK_FPR = 0.05
REVIEW_FPR = 0.20

# Regularisation grid for the Platt fit, selected by cross-validated Brier
# score. This matters more than it looks: the raw signal only spans roughly
# 0.43-0.76, so a strongly regularised fit cannot find the slope it needs and
# collapses every score into a narrow band around 60 - which is exactly the
# original bug. Selecting C on a proper scoring rule fixes it without letting
# anyone hand-pick a flattering number.
PLATT_C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

NON_HUMAN_CLASSES = ("ai", "hybrid")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _non_human_probability(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """P(ai) + P(hybrid) — the raw, uncalibrated 'not human' mass."""
    index = {label: i for i, label in enumerate(classes)}
    total = np.zeros(probabilities.shape[0])
    for label in NON_HUMAN_CLASSES:
        if label in index:
            total += probabilities[:, index[label]]
    return total


def calibrate() -> dict:
    texts, labels = load_dataset(DATASET_PATH)
    y = np.asarray(labels)
    n_folds = 6

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Out-of-fold probabilities: every prediction comes from a model that never
    # saw that sample. This is the only defensible basis for calibration.
    oof_proba = cross_val_predict(
        build_pipeline(), texts, y, cv=splitter, method="predict_proba"
    )
    classes = np.unique(y)
    oof_pred = classes[oof_proba.argmax(axis=1)]

    raw = _non_human_probability(oof_proba, classes)
    is_non_human = np.isin(y, NON_HUMAN_CLASSES).astype(int)

    # Platt scaling: map the raw mass onto a probability that actually behaves
    # like one, fitted against the out-of-fold ground truth.
    feature = raw.reshape(-1, 1)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    brier_by_c = {
        c: float(
            -cross_val_score(
                LogisticRegression(max_iter=10000, C=c),
                feature,
                is_non_human,
                cv=inner_cv,
                scoring="neg_brier_score",
            ).mean()
        )
        for c in PLATT_C_GRID
    }
    best_c = min(brier_by_c, key=brier_by_c.get)

    platt = LogisticRegression(max_iter=10000, C=best_c)
    platt.fit(feature, is_non_human)
    a = float(platt.coef_[0][0])
    b = float(platt.intercept_[0])

    def calibrated(value: float) -> float:
        return _sigmoid(a * value + b) * 100.0

    human_raw = np.sort(raw[is_non_human == 0])

    def cut_at_fpr(fpr: float) -> float:
        """Raw threshold that mislabels at most `fpr` of known human samples."""
        if human_raw.size == 0:
            return 0.5
        return float(np.quantile(human_raw, 1.0 - fpr))

    high_cut = round(calibrated(cut_at_fpr(HIGH_RISK_FPR)), 1)
    review_cut = round(calibrated(cut_at_fpr(REVIEW_FPR)), 1)
    # Bands must stay ordered even if the quantiles collide on a small sample.
    review_cut = min(review_cut, high_cut - 0.1)

    correct = oof_pred == y
    human_scores = [calibrated(v) for v in raw[is_non_human == 0]]
    non_human_scores = [calibrated(v) for v in raw[is_non_human == 1]]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, oof_pred, average="macro", zero_division=0
    )
    # Per-fold spread, straight from the out-of-fold predictions. This is the
    # number that shows how unstable a 252-sample estimate really is.
    fold_f1 = [
        round(
            float(f1_score(y[test], oof_pred[test], average="macro", zero_division=0)),
            4,
        )
        for _, test in splitter.split(texts, y)
    ]

    calibration = {
        "version": 1,
        "source_dataset": DATASET_PATH.name,
        # The detector checks this at runtime. A model pickled by one
        # scikit-learn version and scored under another shifted the raw signal
        # by up to 0.04, which the steep Platt slope turns into a double-digit
        # score change - so a mismatch must not pass silently.
        "sklearn_version": sklearn.__version__,
        "n_samples": int(len(texts)),
        "cv_folds": n_folds,
        "platt": {"a": a, "b": b, "C": best_c, "brier_by_C": brier_by_c},
        "bands": {"review": review_cut, "high": high_cut},
        "abstain": {"min_words": MIN_WORDS},
        "target_false_positive_rate": {
            "high_risk": HIGH_RISK_FPR,
            "needs_review": REVIEW_FPR,
        },
        "out_of_fold_metrics": {
            "macro_precision": round(float(precision), 4),
            "macro_recall": round(float(recall), 4),
            "macro_f1": round(float(f1), 4),
            "accuracy": round(float(correct.mean()), 4),
            "per_fold_macro_f1": fold_f1,
            "fold_macro_f1_spread": round(
                float(max(fold_f1) - min(fold_f1)), 4
            ),
        },
        "separability": {
            "auc_human_vs_generated": round(
                float(roc_auc_score(is_non_human, raw)), 4
            ),
            "human_score_median": round(float(np.median(human_scores)), 1),
            "human_score_p95": round(float(np.quantile(human_scores, 0.95)), 1),
            "generated_score_median": round(float(np.median(non_human_scores)), 1),
            "generated_score_p05": round(
                float(np.quantile(non_human_scores, 0.05)), 1
            ),
        },
        "caveats": [
            "Calibrated on 252 diverse academic samples (median ~21 words). Treat every "
            "number here as indicative, not authoritative.",
            "Calibration is fitted on out-of-fold predictions from pipelines "
            "refit per fold, so it approximates - but is not identical to - "
            "the behaviour of the single shipped model.",
            "Scores are evidence for a human reviewer to weigh, never grounds "
            "for an automated accusation.",
        ],
    }

    CALIBRATION_PATH.write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    return calibration


if __name__ == "__main__":
    result = calibrate()
    print(json.dumps(result, indent=2))
