"""Scoring weights and thresholds."""

from sufficiency_scorer.models import Dimension

# Weight per dimension (sum = 1.0)
# Emotion is highest — it's the gatekeeper
WEIGHTS: dict[Dimension, float] = {
    Dimension.EMOTION: 0.20,
    Dimension.EQ: 0.10,           # behavioral + valence + distress (zero-cost)
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

# Emotion gate: if emotion is not activated, cap score here
EMOTION_GATE_CAP = 0.45

# Number of activated dimensions needed for 100%
ACTIVATION_TARGET = 7

# Minimum confidence to count as activated
MIN_CONFIDENCE = 0.15

# Ready threshold
READY_THRESHOLD = 0.95

# Parallelization groups — detectors within each group can run concurrently
# Groups are ordered by dependency (group 0 first, then group 1, etc.)
PARALLEL_GROUPS: list[list[Dimension]] = [
    # Group 0: zero-cost, no LLM (run first, <10ms)
    [Dimension.EQ],
    # Group 1: all LLM-based detectors (run in parallel, ~1-2s)
    [
        Dimension.EMOTION,
        Dimension.CONFLICT,
        Dimension.HUMOR,
        Dimension.MBTI,
        Dimension.FRAGILITY,
        Dimension.LOVE_LANGUAGE,
        Dimension.CONNECTION_RESPONSE,
        Dimension.CHARACTER,
        Dimension.COMMUNICATION_DNA,
        Dimension.SOULGRAPH,
    ],
]
