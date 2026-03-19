"""EQ adapter — behavioral features + valence + distress (zero LLM cost)."""

import re
import math

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

# Behavioral feature extraction (pure text analysis, no LLM)
SELF_REF_WORDS = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
OTHER_REF_WORDS = {"you", "your", "yours", "yourself", "they", "them", "their"}
HEDGE_WORDS = {"maybe", "perhaps", "might", "probably", "possibly", "sort of", "kind of", "i think", "i guess"}
ABSOLUTIST_WORDS = {"always", "never", "completely", "totally", "absolutely", "nothing", "everything", "everyone", "nobody"}
NEG_EMOTION_WORDS = {"angry", "sad", "afraid", "scared", "anxious", "worried", "frustrated", "depressed", "hopeless", "tired", "exhausted", "stressed", "hurt", "lonely", "miserable", "upset", "overwhelmed"}
POS_EMOTION_WORDS = {"happy", "glad", "excited", "grateful", "hopeful", "proud", "confident", "joyful", "peaceful", "content", "love", "wonderful", "amazing", "great"}


def extract_behavioral(text: str) -> dict:
    """Extract behavioral features from text — zero cost, pure regex."""
    words = text.lower().split()
    word_count = len(words)
    if word_count == 0:
        return {
            "self_ref": 0.0, "other_ref": 0.0, "question_ratio": 0.0,
            "hedging_ratio": 0.0, "absolutist_ratio": 0.0,
            "exclamation_ratio": 0.0, "neg_emotion_ratio": 0.0,
            "pos_emotion_ratio": 0.0, "words": 0,
        }

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    questions = sum(1 for s in sentences if s.strip().endswith("?") or text.count("?") > 0)
    # Recount questions from original text
    questions = text.count("?")

    word_set = set(words)

    self_ref = len([w for w in words if w in SELF_REF_WORDS]) / word_count
    other_ref = len([w for w in words if w in OTHER_REF_WORDS]) / word_count
    hedging = len([w for w in words if w in HEDGE_WORDS]) / word_count
    absolutist = len([w for w in words if w in ABSOLUTIST_WORDS]) / word_count
    neg_emotion = len([w for w in words if w in NEG_EMOTION_WORDS]) / word_count
    pos_emotion = len([w for w in words if w in POS_EMOTION_WORDS]) / word_count
    exclamation = text.count("!") / sentence_count
    question_ratio = questions / sentence_count

    return {
        "self_ref": round(self_ref, 4),
        "other_ref": round(other_ref, 4),
        "question_ratio": round(question_ratio, 4),
        "hedging_ratio": round(hedging, 4),
        "absolutist_ratio": round(absolutist, 4),
        "exclamation_ratio": round(exclamation, 4),
        "neg_emotion_ratio": round(neg_emotion, 4),
        "pos_emotion_ratio": round(pos_emotion, 4),
        "words": word_count,
    }


def compute_valence(features: dict) -> float:
    """Valence from behavioral features. Negative = distress. Range ~ -1.0 to 1.0."""
    pos = features.get("pos_emotion_ratio", 0.0)
    neg = features.get("neg_emotion_ratio", 0.0)
    hedge = features.get("hedging_ratio", 0.0)
    absolutist = features.get("absolutist_ratio", 0.0)
    # Simple linear model (Ridge-inspired)
    valence = (pos * 2.0) - (neg * 2.5) - (absolutist * 0.5) - (hedge * 0.3)
    return max(-1.0, min(1.0, valence))


def compute_distress(features: dict) -> float:
    """Distress score from behavioral features. Range 0.0 to 1.0."""
    neg = features.get("neg_emotion_ratio", 0.0)
    self_ref = features.get("self_ref", 0.0)
    question = features.get("question_ratio", 0.0)
    absolutist = features.get("absolutist_ratio", 0.0)
    # High self-reference + negative emotion + absolutist thinking = distress
    raw = (neg * 3.0) + (self_ref * 0.8) + (absolutist * 1.5) + (question * 0.3)
    return max(0.0, min(1.0, raw))


class EQAdapter(DetectorAdapter):
    """Behavioral features + valence + distress — zero LLM cost.

    Activation: word count >= 5 AND (any non-zero signal detected).
    Confidence: based on how much signal is present.
    """

    dimension = Dimension.EQ

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        features = extract_behavioral(text)
        valence = compute_valence(features)
        distress = compute_distress(features)

        word_count = features["words"]
        if word_count < 3:
            return self._make_result(0.0, detail={"features": features})

        # Signal strength: any meaningful behavioral pattern
        signal_sum = (
            features["self_ref"]
            + features["neg_emotion_ratio"]
            + features["pos_emotion_ratio"]
            + features["hedging_ratio"]
            + features["absolutist_ratio"]
            + abs(valence) * 0.5
            + distress * 0.5
        )
        confidence = min(1.0, signal_sum * 2.0)

        return self._make_result(
            confidence=confidence,
            detail={
                "features": features,
                "valence": round(valence, 4),
                "distress": round(distress, 4),
            },
        )
