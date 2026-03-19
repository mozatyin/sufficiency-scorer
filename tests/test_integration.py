"""Integration tests for real detector adapters.

These tests call the actual detector projects (emotion-detector, conflict-detector,
fragility-detector) which make real Anthropic API calls. They are skipped when:
  - ANTHROPIC_API_KEY is not set
  - The detector project directory does not exist

Run explicitly: .venv/bin/pytest tests/test_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio  # noqa: F401 — ensures pytest-asyncio is available

from sufficiency_scorer.models import Dimension

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
EMOTION_PATH = Path.home() / "emotion-detector"
CONFLICT_PATH = Path.home() / "conflict-detector"
FRAGILITY_PATH = Path.home() / "fragility-detector"

skip_no_api = pytest.mark.skipif(
    not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set"
)
skip_no_emotion = pytest.mark.skipif(
    not EMOTION_PATH.exists(), reason="~/emotion-detector not found"
)
skip_no_conflict = pytest.mark.skipif(
    not CONFLICT_PATH.exists(), reason="~/conflict-detector not found"
)
skip_no_fragility = pytest.mark.skipif(
    not FRAGILITY_PATH.exists(), reason="~/fragility-detector not found"
)

# Sample texts that should trigger each detector
EMOTION_TEXT = (
    "I'm so frustrated and angry right now. My coworker took credit for my "
    "project in front of everyone and I feel completely humiliated and betrayed."
)
CONFLICT_TEXT = (
    "Every time I try to bring up the issue he just shuts down and refuses to "
    "talk. I've tried compromising but he won't meet me halfway. I'm starting "
    "to think I should just avoid the topic entirely."
)
FRAGILITY_TEXT = (
    "I don't know, maybe I'm just not good enough. Everyone else seems to have "
    "it figured out and I'm just pretending. Sometimes I wonder if anyone would "
    "notice if I just stopped trying."
)


# ---------------------------------------------------------------------------
# Emotion adapter
# ---------------------------------------------------------------------------

@skip_no_api
@skip_no_emotion
@pytest.mark.asyncio
async def test_emotion_adapter_returns_result():
    """EmotionAdapter should return a DetectorResult with top_emotions."""
    from sufficiency_scorer.detectors.emotion import EmotionAdapter

    adapter = EmotionAdapter()
    result = await adapter.detect(EMOTION_TEXT)

    assert result.dimension == Dimension.EMOTION
    assert 0.0 <= result.confidence <= 1.0
    assert "top_emotions" in result.detail
    assert isinstance(result.detail["top_emotions"], list)


@skip_no_api
@skip_no_emotion
@pytest.mark.asyncio
async def test_emotion_adapter_activated_on_strong_text():
    """Strong emotional text should activate the detector."""
    from sufficiency_scorer.detectors.emotion import EmotionAdapter

    adapter = EmotionAdapter()
    result = await adapter.detect(EMOTION_TEXT)

    assert result.activated is True
    assert result.confidence > 0.2


# ---------------------------------------------------------------------------
# Conflict adapter
# ---------------------------------------------------------------------------

@skip_no_api
@skip_no_conflict
@pytest.mark.asyncio
async def test_conflict_adapter_returns_result():
    """ConflictAdapter should return a DetectorResult with style scores."""
    from sufficiency_scorer.detectors.conflict import ConflictAdapter

    adapter = ConflictAdapter()
    result = await adapter.detect(CONFLICT_TEXT)

    assert result.dimension == Dimension.CONFLICT
    assert 0.0 <= result.confidence <= 1.0
    assert "styles" in result.detail
    assert isinstance(result.detail["styles"], dict)


@skip_no_api
@skip_no_conflict
@pytest.mark.asyncio
async def test_conflict_adapter_activated_on_conflict_text():
    """Text with clear conflict patterns should activate the detector."""
    from sufficiency_scorer.detectors.conflict import ConflictAdapter

    adapter = ConflictAdapter()
    result = await adapter.detect(CONFLICT_TEXT)

    assert result.activated is True
    assert result.confidence > 0.2


# ---------------------------------------------------------------------------
# Fragility adapter
# ---------------------------------------------------------------------------

@skip_no_api
@skip_no_fragility
@pytest.mark.asyncio
async def test_fragility_adapter_returns_result():
    """FragilityAdapter should return a DetectorResult with pattern info."""
    from sufficiency_scorer.detectors.fragility import FragilityAdapter

    adapter = FragilityAdapter()
    result = await adapter.detect(FRAGILITY_TEXT)

    assert result.dimension == Dimension.FRAGILITY
    assert 0.0 <= result.confidence <= 1.0
    assert "pattern" in result.detail
    assert "pattern_scores" in result.detail
    assert isinstance(result.detail["pattern_scores"], dict)


@skip_no_api
@skip_no_fragility
@pytest.mark.asyncio
async def test_fragility_adapter_pattern_is_string():
    """Pattern should be a plain string (not an Enum object)."""
    from sufficiency_scorer.detectors.fragility import FragilityAdapter

    adapter = FragilityAdapter()
    result = await adapter.detect(FRAGILITY_TEXT)

    pattern = result.detail.get("pattern")
    if pattern is not None:
        assert isinstance(pattern, str)
        assert pattern in ("open", "defensive", "masked", "denial")


@skip_no_api
@skip_no_fragility
@pytest.mark.asyncio
async def test_fragility_adapter_activated_on_vulnerable_text():
    """Vulnerable text should activate the fragility detector."""
    from sufficiency_scorer.detectors.fragility import FragilityAdapter

    adapter = FragilityAdapter()
    result = await adapter.detect(FRAGILITY_TEXT)

    assert result.activated is True
    assert result.confidence > 0.2


# ---------------------------------------------------------------------------
# Cross-adapter: all three run concurrently
# ---------------------------------------------------------------------------

@skip_no_api
@skip_no_emotion
@skip_no_conflict
@skip_no_fragility
@pytest.mark.asyncio
async def test_all_three_adapters_run_concurrently():
    """All three adapters should be able to run via asyncio.gather()."""
    import asyncio

    from sufficiency_scorer.detectors.emotion import EmotionAdapter
    from sufficiency_scorer.detectors.conflict import ConflictAdapter
    from sufficiency_scorer.detectors.fragility import FragilityAdapter

    text = (
        "I'm really upset because my partner and I keep fighting about money. "
        "I feel like I can't be honest about how scared I am. Every time I try "
        "to talk about it he gets defensive and I just shut down."
    )

    emotion = EmotionAdapter()
    conflict = ConflictAdapter()
    fragility = FragilityAdapter()

    results = await asyncio.gather(
        emotion.detect(text),
        conflict.detect(text),
        fragility.detect(text),
    )

    assert len(results) == 3
    dims = {r.dimension for r in results}
    assert dims == {Dimension.EMOTION, Dimension.CONFLICT, Dimension.FRAGILITY}

    for r in results:
        assert 0.0 <= r.confidence <= 1.0
        assert isinstance(r.detail, dict)
