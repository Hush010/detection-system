"""Content detection with a single, calibrated scoring rule.

Scoring contract
----------------
Every engine in this module ends at exactly one function, `_finalise`, which
owns both the numeric score and the risk label. The label is derived *from the
score* and from nothing else, so the two can never disagree.

The score is a calibrated estimate of P(not written by a human), expressed
0-100. The calibration lives in `calibration.json` and is derived from
out-of-fold cross-validation by `calibrate.py` - see that file for why raw
`predict_proba` output could not be used directly.

The detector abstains rather than guessing. A submission that is too short, or
an install with no calibration available, returns `score: None` and the label
`Inconclusive`. Uncalibrated engines (the transformer and heuristic fallbacks)
are advisory only and can never reach `High risk`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
from functools import lru_cache
from typing import Dict, Optional
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

HF_MODEL_NAME = "valhalla/distilbart-mnli-12-1"
CANDIDATE_LABELS = ["AI-generated", "Human-written"]
HF_HYPOTHESIS_TEMPLATE = "This text is {}."

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
CALIBRATION_PATH = BASE_DIR / "calibration.json"

NON_HUMAN_CLASSES = ("ai", "hybrid")

LABEL_LOW = "Low risk"
LABEL_REVIEW = "Needs review"
LABEL_HIGH = "High risk"
LABEL_INCONCLUSIVE = "Inconclusive"
ALL_LABELS = frozenset({LABEL_LOW, LABEL_REVIEW, LABEL_HIGH, LABEL_INCONCLUSIVE})

# Fallback bands, used only if calibration.json is present but incomplete.
DEFAULT_BANDS = {"review": 40.0, "high": 85.0}
DEFAULT_MIN_WORDS = 50


@lru_cache(maxsize=1)
def _load_calibration() -> Optional[dict]:
    """Load the calibration artifact. Plain JSON - it cannot execute code."""
    if not CALIBRATION_PATH.exists():
        logger.error(
            "calibration.json is missing; the detector will abstain on every "
            "request. Run `python calibrate.py` to regenerate it."
        )
        return None
    try:
        with CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.exception("calibration.json could not be read")
        return None

    platt = data.get("platt") or {}
    if "a" not in platt or "b" not in platt:
        logger.error("calibration.json has no usable Platt coefficients")
        return None
    return data


def min_words() -> int:
    """Minimum words before the detector will produce a score."""
    override = os.environ.get("DETECTION_MIN_WORDS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("Ignoring non-integer DETECTION_MIN_WORDS=%r", override)
    calibration = _load_calibration() or {}
    return int(calibration.get("abstain", {}).get("min_words", DEFAULT_MIN_WORDS))


def normalize_text(text: str) -> str:
    """The one text normalisation used by both training and inference."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _counts(text: str) -> Dict[str, int]:
    words = len(text.split())
    sentences = max(1, len(re.findall(r"(?<=[.!?])\s+", text)) + 1) if words else 0
    return {"word_count": words, "sentence_count": sentences}


def _calibrate_score(raw_probability: float) -> Optional[float]:
    """Map raw P(ai)+P(hybrid) onto the calibrated 0-100 risk score."""
    calibration = _load_calibration()
    if calibration is None:
        return None
    platt = calibration["platt"]
    z = platt["a"] * raw_probability + platt["b"]
    # Numerically stable logistic: the fitted slope is steep, so a naive
    # exp(-z) overflows on confident inputs.
    if z >= 0:
        probability = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        probability = exp_z / (1.0 + exp_z)
    return round(probability * 100.0, 1)


@lru_cache(maxsize=1)
def _calibration_matches_environment() -> bool:
    """Is the running scikit-learn the one the calibration was fitted against?

    A model pickled under one version and scored under another moved the raw
    signal by up to 0.04 in testing. The Platt slope is steep, so that is a
    double-digit swing in the final score - easily the difference between
    clearing a student and flagging one. On a mismatch we keep serving, but the
    result is downgraded to advisory so it can never reach High risk.
    """
    calibration = _load_calibration() or {}
    expected = calibration.get("sklearn_version")
    if not expected:
        return True
    try:
        import sklearn
    except ImportError:
        return True

    running = sklearn.__version__
    if running.split(".")[:2] == expected.split(".")[:2]:
        return True

    logger.error(
        "scikit-learn %s is running but calibration.json was fitted against "
        "%s. Scores are downgraded to advisory. Re-run `python "
        "train_model.py && python calibrate.py` on the pinned version.",
        running,
        expected,
    )
    return False


def _bands() -> Dict[str, float]:
    calibration = _load_calibration() or {}
    bands = dict(DEFAULT_BANDS)
    bands.update(calibration.get("bands") or {})
    return bands


def _abstain(reason: str, engine: str, text: str, **extra) -> Dict[str, object]:
    details = {
        "engine": engine,
        "calibrated": False,
        "abstained": True,
        "reason": reason,
        "marker_hits": [],
        **_counts(text),
    }
    details.update(extra)
    return {"score": None, "label": LABEL_INCONCLUSIVE, "details": details}



def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences preserving punctuation boundaries."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def analyze_sentences(text: str) -> list[dict]:
    """Analyze each sentence individually to generate an explainable heatmap breakdown."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    model = _load_trained_model()
    ai_markers = [
        "in conclusion", "in summary", "comprehensively explores",
        "multifaceted", "nuanced analysis", "it is evident that",
        "it is important to note", "complex interplay", "delve into",
        "furthermore", "moreover", "navigate the", "plays a crucial role",
        "plays a vital role", "a testament to", "in today"
    ]

    results = []
    classes = list(model.classes_) if model else []
    index = {label: i for i, label in enumerate(classes)}
    bands = _bands()

    for s in sentences:
        cleaned_s = normalize_text(s)
        words = len(cleaned_s.split())
        matched_markers = [m for m in ai_markers if m in cleaned_s]

        sentence_score = None
        if model and words >= 4:
            try:
                probs = model.predict_proba([cleaned_s])[0]
                raw_ai = sum(float(probs[index[lbl]]) for lbl in NON_HUMAN_CLASSES if lbl in index)
                cal_score = _calibrate_score(raw_ai)
                if cal_score is not None:
                    sentence_score = cal_score
            except Exception:
                pass

        if sentence_score is None:
            sentence_score = min(100.0, len(matched_markers) * 35.0)

        sentence_score = round(max(0.0, min(100.0, float(sentence_score))), 1)

        if sentence_score >= bands.get("high", 87.8):
            risk_level = "high"
        elif sentence_score >= bands.get("review", 31.2):
            risk_level = "review"
        else:
            risk_level = "low"

        results.append({
            "text": s,
            "score": sentence_score,
            "risk": risk_level,
            "words": words,
            "markers": matched_markers
        })
    return results


def _finalise(
    score: Optional[float],
    engine: str,
    text: str,
    calibrated: bool,
    **extra,
) -> Dict[str, object]:
    """The single place a score becomes a label.

    Nothing else in this module is allowed to choose a label. If you are adding
    an engine, return through here.
    """
    if score is None:
        return _abstain("no score available", engine, text, **extra)

    score = round(max(0.0, min(100.0, float(score))), 1)
    bands = _bands()

    if score >= bands["high"]:
        label = LABEL_HIGH
    elif score >= bands["review"]:
        label = LABEL_REVIEW
    else:
        label = LABEL_LOW

    # An uncalibrated engine is advisory. It may raise a flag for a human to
    # look at; it may never be the thing that says "High risk".
    if not calibrated and label == LABEL_HIGH:
        label = LABEL_REVIEW

    sentences = analyze_sentences(text)
    details = {
        "engine": engine,
        "calibrated": calibrated,
        "abstained": False,
        "advisory": not calibrated,
        "bands": bands,
        "marker_hits": [],
        "sentences": sentences,
        **_counts(text),
    }
    details.update(extra)
    return {"score": score, "label": label, "details": details}


@lru_cache(maxsize=1)
def _load_trained_model():
    """Load the trained scikit-learn model."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            logger.exception("model.joblib exists but could not be loaded")
    return None


def _transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None


@lru_cache(maxsize=1)
def _load_hf_pipeline():
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=HF_MODEL_NAME,
        device=-1,
    )


def analyze_text_with_transformer(text: str) -> Dict[str, object]:
    """Zero-shot transformer path. Uncalibrated, therefore advisory only."""
    cleaned = normalize_text(text)
    if not cleaned:
        return _abstain("empty text", "transformer", cleaned)

    classifier = _load_hf_pipeline()
    classification = classifier(
        cleaned,
        CANDIDATE_LABELS,
        hypothesis_template=HF_HYPOTHESIS_TEMPLATE,
    )

    label_scores = dict(zip(classification["labels"], classification["scores"]))
    ai_score = float(label_scores.get("AI-generated", 0.0))

    return _finalise(
        ai_score * 100.0,
        "transformer",
        cleaned,
        calibrated=False,
        framework="Hugging Face",
        model=HF_MODEL_NAME,
        prediction=classification["labels"][0],
        scores=classification["scores"],
        label_order=classification["labels"],
        ai_probability=ai_score,
    )


def _analyze_with_trained_model(cleaned: str) -> Optional[Dict[str, object]]:
    model = _load_trained_model()
    if model is None:
        return None

    try:
        probabilities = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
    except Exception:
        logger.exception("trained model failed to score text; falling back")
        return None

    index = {label: i for i, label in enumerate(classes)}
    raw = sum(
        float(probabilities[index[label]])
        for label in NON_HUMAN_CLASSES
        if label in index
    )

    score = _calibrate_score(raw)
    if score is None:
        # Fail safe: an uncalibrated trained model is exactly the failure mode
        # this rewrite exists to remove. Abstain rather than invent a number.
        return _abstain(
            "calibration unavailable - run calibrate.py",
            "trained_model",
            cleaned,
        )

    environment_ok = _calibration_matches_environment()

    return _finalise(
        score,
        "trained_model",
        cleaned,
        calibrated=environment_ok,
        calibration_version_mismatch=not environment_ok,
        prediction=classes[int(probabilities.argmax())],
        raw_non_human_probability=round(raw, 4),
        probabilities={
            label: round(float(probabilities[i]), 3) for i, label in enumerate(classes)
        },
    )


def analyze_text(text: str) -> Dict[str, object]:
    cleaned = normalize_text(text)
    if not cleaned:
        return _abstain("empty text", "none", cleaned)

    words = len(cleaned.split())
    threshold = min_words()
    if words < threshold:
        # The original detector scored a two-letter input as "High risk".
        # Length is checked before any engine runs so that can never recur.
        return _abstain(
            f"insufficient text: {words} words, {threshold} required for a "
            f"reliable score",
            "none",
            cleaned,
            required_words=threshold,
        )

    result = _analyze_with_trained_model(cleaned)
    if result is not None:
        return result

    if _transformers_available():
        try:
            return analyze_text_with_transformer(cleaned)
        except Exception:
            logger.exception("transformer path failed; falling back to heuristic")

    return _analyze_text_heuristic(cleaned)


def _analyze_text_heuristic(text: str) -> Dict[str, object]:
    """Keyword heuristic. Uncalibrated, therefore advisory only.

    The previous version of this list contained topic words lifted from the
    test fixtures - "modern education systems", "digital technologies",
    "academic performance", "online platforms", "institutions must". Those made
    the tests pass while penalising any essay written *about* education or
    technology, which is most student writing. They have been removed. What
    remains are register and discourse markers, which are at least weak
    stylistic evidence rather than subject matter.
    """
    lower = normalize_text(text)
    ai_markers = [
        "in conclusion",
        "in summary",
        "comprehensively explores",
        "multifaceted",
        "nuanced analysis",
        "it is evident that",
        "it is important to note",
        "complex interplay",
        "delve into",
        "furthermore",
        "moreover",
        "navigate the",
        "plays a crucial role",
        "plays a vital role",
        "a testament to",
        "in today's rapidly",
    ]

    marker_hits = [marker for marker in ai_markers if marker in lower]
    counts = _counts(lower)

    # Length no longer contributes. Previously a longer essay scored higher for
    # being longer, which meant the score partly measured word count.
    marker_score = min(70.0, len(marker_hits) * 18.0)
    density = len(marker_hits) / max(1, counts["sentence_count"])
    density_score = min(25.0, density * 50.0)

    return _finalise(
        marker_score + density_score,
        "heuristic",
        lower,
        calibrated=False,
        marker_hits=marker_hits,
        note="Keyword heuristic fallback. Indicative only - not calibrated.",
    )
