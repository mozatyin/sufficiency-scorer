"""Scoring thresholds and parallelization config."""

from sufficiency_scorer.models import Dimension

# === Insight Gate ===
INSIGHT_THRESHOLD = 3
MIN_INSIGHT_QUALITY_FOR_READY = 2  # InsightQuality.MEDIUM or higher

# === Word Gate ===
MIN_WORDS = 40

# === Confidence Thresholds ===
MIN_CONFIDENCE = 0.15
MIN_INSIGHT_CONFIDENCE = 0.4

# === Parallelization ===
PARALLEL_GROUPS: list[list[Dimension]] = [
    [Dimension.EQ],
    [
        Dimension.EMOTION, Dimension.CONFLICT, Dimension.HUMOR,
        Dimension.MBTI, Dimension.FRAGILITY, Dimension.LOVE_LANGUAGE,
        Dimension.CONNECTION_RESPONSE, Dimension.CHARACTER,
        Dimension.COMMUNICATION_DNA, Dimension.SOULGRAPH,
    ],
]

# === Legacy compat (removed in scorer.py rewrite) ===
WEIGHTS: dict[Dimension, float] = {
    Dimension.EMOTION: 0.20,
    Dimension.EQ: 0.10,
    Dimension.FRAGILITY: 0.10,
    Dimension.CONFLICT: 0.10,
    Dimension.HUMOR: 0.08,
    Dimension.MBTI: 0.08,
    Dimension.LOVE_LANGUAGE: 0.08,
    Dimension.COMMUNICATION_DNA: 0.08,
    Dimension.CONNECTION_RESPONSE: 0.06,
    Dimension.CHARACTER: 0.06,
    Dimension.SOULGRAPH: 0.06,
}
EMOTION_GATE_CAP = 0.45
ACTIVATION_TARGET = 7
READY_THRESHOLD = 0.95
