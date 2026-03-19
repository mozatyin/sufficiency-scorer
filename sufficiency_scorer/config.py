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
