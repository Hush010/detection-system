import re
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib


DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    texts = [preprocess_text(item["text"]) for item in raw]
    labels = [item["label"] for item in raw]
    return texts, labels


def train_and_evaluate():
    texts, labels = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, multi_class='multinomial'),
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "classification_report": report,
    }

    joblib.dump(pipeline, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
