import importlib.util
import re
from functools import lru_cache
from typing import Dict, Optional

HF_MODEL_NAME = "valhalla/distilbart-mnli-12-1"
CANDIDATE_LABELS = ["AI-generated", "Human-written"]
HF_HYPOTHESIS_TEMPLATE = "This text is {}."


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
            "details": {"engine": "transformer", "reason": "empty text"},
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
        },
    }


def analyze_text(text: str) -> Dict[str, object]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return {
            "score": 0.0,
            "label": "Low risk",
            "details": {"engine": "heuristic", "reason": "empty text"},
        }

    if _transformers_available():
        try:
            return analyze_text_with_transformer(cleaned)
        except Exception:
            pass

    lower = cleaned.lower()
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

    word_count = len(cleaned.split())
    sentence_count = max(1, len(re.findall(r"(?<=[.!?])\s+", cleaned)) + 1)

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
