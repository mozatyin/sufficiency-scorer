"""Insight quality evaluation framework.

Scores insights on two axes:
1. Specificity: Is this insight specific to THIS user, or a generic platitude?
2. Reframe quality: Does it transform the signal into something positive and surprising?

Used for:
- Validating that our reframe templates are actually good
- Testing new cross-patterns before deploying
- Monitoring insight quality over time
"""

import re
from pydantic import BaseModel, Field

from sufficiency_scorer.models import InsightCandidate, InsightQuality


class EvalResult(BaseModel):
    """Evaluation of a single insight."""
    specificity: float = Field(ge=0.0, le=1.0)
    reframe_quality: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class BatchEvalResult(BaseModel):
    """Evaluation of a batch of insights."""
    total: int = 0
    avg_specificity: float = 0.0
    avg_reframe_quality: float = 0.0
    avg_overall: float = 0.0
    bloom_worthy: bool = False
    per_insight: list[EvalResult] = Field(default_factory=list)


# Words that indicate generic platitudes
PLATITUDE_INDICATORS = {
    "good person", "strong person", "great", "amazing", "wonderful",
    "special", "unique", "important", "valuable", "worthy",
}

# Words/phrases that indicate negative framing (should be reframed positively)
NEGATIVE_INDICATORS = {
    "anxious", "depressed", "worried", "stressed", "afraid",
    "weak", "broken", "damaged", "toxic", "dysfunctional",
    "very anxious", "very worried", "really stressed",
}

# Positive reframe indicators — suggest transformation happened
POSITIVE_REFRAME_INDICATORS = {
    "courage", "strength", "aware", "honest", "deep", "care",
    "protect", "resilience", "wisdom", "clarity", "intelligence",
    "brave", "real", "genuine", "open", "growth", "backbone",
    "superpower", "sharp", "tuned", "strategic",
}

# Signal leakage — when the reframe just repeats the signal name
SIGNAL_WORDS = {
    "frustrated", "frustration", "angry", "anger", "sad", "sadness",
    "anxious", "anxiety", "confused", "confusion", "afraid", "fear",
    "avoid", "avoidance", "confront", "confrontation",
}


class InsightEvaluator:
    """Evaluates insight quality on specificity and reframe transformation."""

    def evaluate(self, insight: InsightCandidate) -> EvalResult:
        specificity = self._score_specificity(insight)
        reframe_quality = self._score_reframe(insight)
        overall = specificity * 0.4 + reframe_quality * 0.6
        flags = self._collect_flags(insight, specificity, reframe_quality)
        return EvalResult(
            specificity=round(specificity, 3),
            reframe_quality=round(reframe_quality, 3),
            overall=round(overall, 3),
            flags=flags,
        )

    def evaluate_batch(self, insights: list[InsightCandidate]) -> BatchEvalResult:
        if not insights:
            return BatchEvalResult(bloom_worthy=False)
        results = [self.evaluate(i) for i in insights]
        avg_spec = sum(r.specificity for r in results) / len(results)
        avg_reframe = sum(r.reframe_quality for r in results) / len(results)
        avg_overall = sum(r.overall for r in results) / len(results)
        # Bloom worthy: avg overall >= 0.5 AND at least 3 insights with overall >= 0.4
        good_count = sum(1 for r in results if r.overall >= 0.4)
        bloom_worthy = avg_overall >= 0.5 and good_count >= 3
        return BatchEvalResult(
            total=len(insights),
            avg_specificity=round(avg_spec, 3),
            avg_reframe_quality=round(avg_reframe, 3),
            avg_overall=round(avg_overall, 3),
            bloom_worthy=bloom_worthy,
            per_insight=results,
        )

    def _score_specificity(self, insight: InsightCandidate) -> float:
        """Score 0-1: how specific is this insight to the user?"""
        reframe = insight.reframe.lower()
        score = 0.5  # baseline

        # Penalty: generic platitudes
        for p in PLATITUDE_INDICATORS:
            if p in reframe:
                score -= 0.3
                break

        # Penalty: just repeating the signal/emotion name
        signal_words_in_reframe = sum(1 for w in SIGNAL_WORDS if w in reframe)
        if signal_words_in_reframe > 0:
            # Check if it's the ONLY content (mere repetition)
            words = reframe.split()
            signal_ratio = signal_words_in_reframe / max(len(words), 1)
            if signal_ratio > 0.2:
                score -= 0.4  # heavy penalty for parrot-back

        # Bonus: cross-dimensional (inherently more specific)
        if len(insight.source_dimensions) > 1:
            score += 0.2

        # Bonus: contains contextual references (work, relationship, etc.)
        context_words = {"work", "boss", "partner", "family", "relationship", "situation", "overtime", "loss"}
        if any(w in reframe for w in context_words):
            score += 0.15

        # Bonus: longer reframes tend to be more specific
        if len(reframe.split()) >= 12:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_reframe(self, insight: InsightCandidate) -> float:
        """Score 0-1: how well does the reframe transform the signal?"""
        reframe = insight.reframe.lower()
        score = 0.5  # baseline

        # Penalty: negative framing
        for neg in NEGATIVE_INDICATORS:
            if neg in reframe:
                score -= 0.3
                break

        # Bonus: positive reframe indicators
        pos_count = sum(1 for p in POSITIVE_REFRAME_INDICATORS if p in reframe)
        score += min(pos_count * 0.1, 0.3)

        # Bonus: uses "you" (addressing the person directly)
        if "you" in reframe.split():
            score += 0.1

        # Penalty: too short (likely low-effort)
        if len(reframe.split()) < 5:
            score -= 0.3

        # Bonus: contains contrast or surprise ("not X, but Y" patterns)
        contrast_patterns = [r"not .+, but", r"instead of", r"rather than", r"more than .+ realize"]
        if any(re.search(p, reframe) for p in contrast_patterns):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _collect_flags(self, insight: InsightCandidate, spec: float, reframe: float) -> list[str]:
        flags = []
        if spec < 0.3:
            flags.append("too_generic")
        if reframe < 0.3:
            flags.append("poor_reframe")
        if not insight.reframe:
            flags.append("empty_reframe")
        r_lower = insight.reframe.lower()
        for w in SIGNAL_WORDS:
            if f"you feel {w}" in r_lower or f"you are {w}" in r_lower:
                flags.append("parrot_back")
                break
        return flags
