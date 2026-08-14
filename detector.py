import importlib.util
import re
from functools import lru_cache
from typing import Dict, Optional
from pathlib import Path
import joblib

HF_MODEL_NAME = "valhalla/distilbart-mnli-12-1"
CANDIDATE_LABELS = ["AI-generated", "Human-written"]
HF_HYPOTHESIS_TEMPLATE = "This text is {}."

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


@lru_cache(maxsize=1)
def _load_trained_model():
    """Load the trained scikit-learn model."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
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
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return {
            "score": 0.0,
            "label": "Low risk",
            "details": {"engine": "transformer", "reason": "empty text", "word_count": 0, "sentence_count": 0, "marker_hits": []},
        }

    classifier = _load_hf_pipeline()
    classification = classifier(
        cleaned,
        CANDIDATE_LABELS,
        hypothesis_template=HF_HYPOTHESIS_TEMPLATE,
    )

    label_scores = dict(zip(classification["labels"], classification["scores"]))
    ai_score = float(label_scores.get("AI-generated", 0.0))
    score = round(ai_score * 100, 1)

    if score >= 70:
        risk_label = "High risk"
    elif score >= 40:
        risk_label = "Medium risk"
    else:
        risk_label = "Low risk"

    word_count = len(cleaned.split())
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", cleaned)) + 1)

    return {
        "score": score,
        "label": risk_label,
        "details": {
            "engine": "transformer",
            "framework": "Hugging Face",
            "model": HF_MODEL_NAME,
            "prediction": classification["labels"][0],
            "scores": classification["scores"],
            "label_order": classification["labels"],
            "ai_probability": ai_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "marker_hits": [],
        },
    }


def analyze_text(text: str) -> Dict[str, object]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return {
            "score": 0.0,
            "label": "Low risk",
            "details": {"engine": "trained_model", "reason": "empty text"},
        }

    # Try trained model first
    model = _load_trained_model()
    if model:
        try:
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            
            # Get probability for AI/hybrid classes
            classes = model.named_steps['logisticregression'].classes_
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Calculate AI risk score
            ai_prob = 0.0
            if 'ai' in class_to_idx:
                ai_prob += probabilities[class_to_idx['ai']]
            if 'hybrid' in class_to_idx:
                ai_prob += probabilities[class_to_idx['hybrid']]
            
            score = round(ai_prob * 100, 1)
            
            # Map predictions to risk labels
            if prediction in ('ai', 'hybrid'):
                risk_label = "High risk"
            else:
                risk_label = "Low risk"
            
            word_count = len(cleaned.split())
            sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", cleaned)) + 1)
            
            return {
                "score": score,
                "label": risk_label,
                "details": {
                    "engine": "trained_model",
                    "prediction": prediction,
                    "probabilities": {classes[i]: round(probabilities[i], 3) for i in range(len(classes))},
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "marker_hits": [],
                },
            }
        except Exception:
            pass  # Fall back to transformer or heuristic

    # Fall back to transformer
    if _transformers_available():
        try:
            return analyze_text_with_transformer(cleaned)
        except Exception:
            pass

    # Fall back to heuristic
    return _analyze_text_heuristic(cleaned)


def _analyze_text_heuristic(text: str) -> Dict[str, object]:
    lower = text.lower()
    ai_markers = [
        "in conclusion",
        "comprehensively explores",
        "multifaceted",
        "nuanced analysis",
        "in summary",
        "overall",
        "furthermore",
        "moreover",
        "it is evident that",
        "this essay",
        "overall,",
        "complex interplay",
        "societal transformation",
        "modern technological advancement",
        "modern education systems",
        "digital technologies",
        "online platforms",
        "academic performance",
        "opportunities and challenges",
        "learning experiences",
        "thoughtful strategies",
        "effective learning",
        "institutions must",
        "knowledge is accessed",
    ]

    marker_hits = [marker for marker in ai_markers if marker in lower]
    abstract_terms = [
        "systems",
        "strategies",
        "opportunities",
        "challenges",
        "experiences",
        "performance",
        "knowledge",
        "institutions",
        "digital",
        "technology",
        "technologies",
        "learning",
    ]
    abstract_hits = [term for term in abstract_terms if term in lower]

    word_count = len(text.split())
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", text)) + 1)

    length_score = min(25, max(0, word_count // 6))
    marker_score = min(60, len(marker_hits) * 18)
    abstract_score = min(20, len(set(abstract_hits)) * 4)
    structure_score = min(15, max(0, sentence_count - 1) * 5)

    score = min(100, length_score + marker_score + abstract_score + structure_score)
    if score >= 75:
        label = "High risk"
    elif score >= 45:
        label = "Medium risk"
    else:
        label = "Low risk"

    return {
        "score": round(score, 1),
        "label": label,
        "details": {
            "engine": "heuristic",
            "marker_hits": marker_hits,
            "word_count": word_count,
            "sentence_count": sentence_count,
        },
    }
