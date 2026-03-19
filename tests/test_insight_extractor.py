"""Tests for the insight extractor — cross-dimensional signal reframing."""

import pytest

from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)


def _dr(dim: Dimension, activated: bool = True, confidence: float = 0.6, detail: dict | None = None) -> DetectorResult:
    """Shorthand factory for DetectorResult."""
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


@pytest.fixture
def extractor():
    return InsightExtractor()


# ===================================================================
# TestSingleDimensionInsights
# ===================================================================

class TestSingleDimensionInsights:
    """Single-dimension pass: each activated detector produces an insight."""

    def test_emotion_produces_insight(self, extractor):
        results = [_dr(Dimension.EMOTION, detail={"top_emotions": [("frustration", 0.55)]})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert insights[0].quality == InsightQuality.MEDIUM
        assert Dimension.EMOTION in insights[0].source_dimensions

    def test_humor_produces_insight(self, extractor):
        results = [_dr(Dimension.HUMOR, detail={"humor_detected": True, "styles": {"self_deprecating": 0.8}})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert Dimension.HUMOR in insights[0].source_dimensions

    def test_conflict_produces_insight(self, extractor):
        results = [_dr(Dimension.CONFLICT, detail={"styles": {"avoid": 0.7, "compromise": 0.4}})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert Dimension.CONFLICT in insights[0].source_dimensions

    def test_fragility_produces_insight(self, extractor):
        results = [_dr(Dimension.FRAGILITY, detail={"pattern": "open", "pattern_scores": {"open": 0.8}})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert Dimension.FRAGILITY in insights[0].source_dimensions

    def test_eq_produces_insight(self, extractor):
        results = [_dr(Dimension.EQ, detail={
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35,
            "distress": 0.43,
        })]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert Dimension.EQ in insights[0].source_dimensions

    def test_inactive_detector_produces_nothing(self, extractor):
        results = [_dr(Dimension.EMOTION, activated=False, detail={"top_emotions": [("anger", 0.9)]})]
        insights = extractor.extract(results)
        assert len(insights) == 0

    def test_low_confidence_detector_produces_nothing(self, extractor):
        """Below MIN_CONFIDENCE (0.15) → ignored."""
        results = [_dr(Dimension.EMOTION, confidence=0.1, detail={"top_emotions": [("anger", 0.9)]})]
        insights = extractor.extract(results)
        assert len(insights) == 0


# ===================================================================
# TestCrossDimensionalInsights
# ===================================================================

class TestCrossDimensionalInsights:
    """Cross-dimensional pass: surprising patterns across detector pairs."""

    def test_emotion_conflict_cross(self, extractor):
        """frustration + avoidance → cross insight."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.6, detail={"top_emotions": [("frustration", 0.55)]}),
            _dr(Dimension.CONFLICT, confidence=0.5, detail={"styles": {"avoid": 0.7}}),
        ]
        insights = extractor.extract(results)
        high = [i for i in insights if i.quality == InsightQuality.HIGH]
        assert len(high) >= 1
        assert Dimension.EMOTION in high[0].source_dimensions
        assert Dimension.CONFLICT in high[0].source_dimensions

    def test_humor_fragility_cross(self, extractor):
        """self-deprecating humor + open fragility → cross insight."""
        results = [
            _dr(Dimension.HUMOR, confidence=0.7, detail={"humor_detected": True, "styles": {"self_deprecating": 0.8}}),
            _dr(Dimension.FRAGILITY, confidence=0.6, detail={"pattern": "open", "pattern_scores": {"open": 0.8}}),
        ]
        insights = extractor.extract(results)
        high = [i for i in insights if i.quality == InsightQuality.HIGH]
        assert len(high) >= 1
        assert set(high[0].source_dimensions) == {Dimension.HUMOR, Dimension.FRAGILITY}

    def test_emotion_eq_cross(self, extractor):
        """strong emotion + high self_ref → cross insight."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.5, detail={"top_emotions": [("sadness", 0.6)]}),
            _dr(Dimension.EQ, confidence=0.5, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.05, "words": 50},
                "valence": -0.1,
                "distress": 0.2,
            }),
        ]
        insights = extractor.extract(results)
        high = [i for i in insights if i.quality == InsightQuality.HIGH]
        assert len(high) >= 1
        dims = set(high[0].source_dimensions)
        assert Dimension.EMOTION in dims
        assert Dimension.EQ in dims

    def test_cross_insights_are_high_quality(self, extractor):
        """All cross-dimensional insights must be HIGH quality."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.6, detail={"top_emotions": [("anger", 0.7)]}),
            _dr(Dimension.CONFLICT, confidence=0.5, detail={"styles": {"confront": 0.8}}),
        ]
        insights = extractor.extract(results)
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        for c in cross:
            assert c.quality == InsightQuality.HIGH

    def test_cross_below_min_insight_confidence_filtered(self, extractor):
        """If both detectors are below MIN_INSIGHT_CONFIDENCE, no cross insight."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.3, detail={"top_emotions": [("frustration", 0.55)]}),
            _dr(Dimension.CONFLICT, confidence=0.3, detail={"styles": {"avoid": 0.7}}),
        ]
        insights = extractor.extract(results)
        high = [i for i in insights if i.quality == InsightQuality.HIGH]
        assert len(high) == 0


# ===================================================================
# TestInsightCount
# ===================================================================

class TestInsightCount:
    """End-to-end count tests — does a scenario produce enough insights?"""

    def test_work_pressure_scenario_produces_3_plus(self, extractor):
        """Rich multi-detector scenario → 3+ MEDIUM+ insights."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.7, detail={"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}),
            _dr(Dimension.EQ, confidence=0.6, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 80},
                "valence": -0.35,
                "distress": 0.43,
            }),
            _dr(Dimension.CONFLICT, confidence=0.5, detail={"styles": {"avoid": 0.7, "compromise": 0.4}}),
            _dr(Dimension.FRAGILITY, confidence=0.5, detail={"pattern": "open", "pattern_scores": {"open": 0.8}}),
            _dr(Dimension.SOULGRAPH, confidence=0.4, detail={"items": 2, "avg_specificity": 0.5}),
        ]
        insights = extractor.extract(results)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]
        assert len(good) >= 3, f"Expected 3+ MEDIUM+ insights, got {len(good)}: {[i.signal for i in good]}"

    def test_gibberish_produces_zero(self, extractor):
        """All inactive → 0 insights."""
        results = [
            _dr(Dimension.EMOTION, activated=False),
            _dr(Dimension.CONFLICT, activated=False),
            _dr(Dimension.HUMOR, activated=False),
            _dr(Dimension.EQ, activated=False),
            _dr(Dimension.FRAGILITY, activated=False),
        ]
        insights = extractor.extract(results)
        assert len(insights) == 0

    def test_single_weak_detector_under_3(self, extractor):
        """One detector alone can't produce 3+ good insights."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.5, detail={"top_emotions": [("sadness", 0.4)]}),
        ]
        insights = extractor.extract(results)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]
        assert len(good) < 3

    def test_empty_results_produces_zero(self, extractor):
        insights = extractor.extract([])
        assert len(insights) == 0


# ===================================================================
# TestReframeQuality
# ===================================================================

class TestReframeQuality:
    """Reframes must be non-empty and must not just repeat the signal name."""

    def test_emotion_reframe_not_empty(self, extractor):
        results = [_dr(Dimension.EMOTION, detail={"top_emotions": [("anger", 0.7)]})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert len(insights[0].reframe) > 0

    def test_emotion_reframe_does_not_repeat_emotion_name(self, extractor):
        for emo in ["frustration", "anger", "sadness", "fear", "anxiety"]:
            results = [_dr(Dimension.EMOTION, detail={"top_emotions": [(emo, 0.6)]})]
            insights = extractor.extract(results)
            assert len(insights) >= 1
            reframe_lower = insights[0].reframe.lower()
            # The reframe must not be "you feel <emotion>" or just the emotion name
            assert reframe_lower != f"you feel {emo}"
            assert reframe_lower != emo

    def test_conflict_reframe_not_empty(self, extractor):
        results = [_dr(Dimension.CONFLICT, detail={"styles": {"collaborate": 0.6}})]
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert len(insights[0].reframe) > 0

    def test_cross_reframe_not_empty(self, extractor):
        results = [
            _dr(Dimension.EMOTION, confidence=0.6, detail={"top_emotions": [("frustration", 0.55)]}),
            _dr(Dimension.CONFLICT, confidence=0.5, detail={"styles": {"avoid": 0.7}}),
        ]
        insights = extractor.extract(results)
        for i in insights:
            assert len(i.reframe) > 10, f"Reframe too short: '{i.reframe}'"

    def test_sorted_by_quality_then_confidence(self, extractor):
        """Insights come out sorted: quality desc, confidence desc."""
        results = [
            _dr(Dimension.EMOTION, confidence=0.6, detail={"top_emotions": [("frustration", 0.55)]}),
            _dr(Dimension.CONFLICT, confidence=0.5, detail={"styles": {"avoid": 0.7}}),
            _dr(Dimension.FRAGILITY, confidence=0.7, detail={"pattern": "open", "pattern_scores": {"open": 0.8}}),
        ]
        insights = extractor.extract(results)
        for i in range(len(insights) - 1):
            a, b = insights[i], insights[i + 1]
            assert (a.quality.value, a.confidence) >= (b.quality.value, b.confidence)
