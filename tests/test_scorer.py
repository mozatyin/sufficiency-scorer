"""Tests for the insight-driven scoring engine."""

import pytest
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality
from sufficiency_scorer.scorer import SufficiencyScorer


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0, detail: dict | None = None) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


def all_inactive() -> list[DetectorResult]:
    return [make_result(dim) for dim in Dimension]


def work_pressure_results() -> list[DetectorResult]:
    """Simulates rich emotional input."""
    base = all_inactive()
    overrides = {
        Dimension.EMOTION: make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
        }),
        Dimension.EQ: make_result(Dimension.EQ, True, 0.6, detail={
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35, "distress": 0.43,
        }),
        Dimension.CONFLICT: make_result(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7, "compromise": 0.4},
        }),
        Dimension.FRAGILITY: make_result(Dimension.FRAGILITY, True, 0.5, detail={
            "pattern": "open", "pattern_scores": {"open": 0.6},
        }),
        Dimension.SOULGRAPH: make_result(Dimension.SOULGRAPH, True, 0.5, detail={
            "items": 2, "avg_specificity": 0.5,
        }),
    }
    return [overrides.get(r.dimension, r) for r in base]


class TestReadyDecision:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_ready_with_rich_input(self, scorer):
        report = scorer.score(work_pressure_results())
        assert report.ready is True
        assert len(report.insights) >= 3

    def test_not_ready_with_nothing(self, scorer):
        report = scorer.score(all_inactive())
        assert report.ready is False
        assert len(report.insights) == 0

    def test_not_ready_with_single_weak_signal(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EQ
            else make_result(Dimension.EQ, True, 0.2, detail={
                "features": {"self_ref": 0.05, "question_ratio": 0.0, "words": 10},
                "valence": -0.1, "distress": 0.1,
            })
            for r in results
        ]
        report = scorer.score(results)
        assert report.ready is False


class TestRingProgress:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_zero_insights_zero_progress(self, scorer):
        report = scorer.score(all_inactive())
        assert report.ring_progress == 0.0

    def test_ready_means_full_progress(self, scorer):
        report = scorer.score(work_pressure_results())
        if report.ready:
            assert report.ring_progress == 1.0

    def test_partial_progress(self, scorer):
        """Single weak emotion → 1 insight → partial progress."""
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION
            else make_result(Dimension.EMOTION, True, 0.4, detail={
                "top_emotions": [("sadness", 0.3)],
            })
            for r in results
        ]
        report = scorer.score(results)
        assert 0.0 < report.ring_progress < 1.0


class TestPromptHint:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_no_insights_tells_more(self, scorer):
        report = scorer.score(all_inactive())
        assert report.prompt_hint == "tell_me_more"

    def test_ready_says_ready(self, scorer):
        report = scorer.score(work_pressure_results())
        assert report.prompt_hint == "ready"

    def test_some_insights_keeps_going(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION
            else make_result(Dimension.EMOTION, True, 0.5, detail={
                "top_emotions": [("sadness", 0.5)],
            })
            for r in results
        ]
        report = scorer.score(results)
        if not report.ready:
            assert report.prompt_hint in ("keep_going", "almost_there", "tell_me_more")


class TestInsightsPassedThrough:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_insights_in_report(self, scorer):
        report = scorer.score(work_pressure_results())
        assert len(report.insights) > 0

    def test_insights_sorted_by_quality(self, scorer):
        report = scorer.score(work_pressure_results())
        if len(report.insights) >= 2:
            for i in range(len(report.insights) - 1):
                assert report.insights[i].quality >= report.insights[i + 1].quality
