"""Tests for the core scoring engine."""

import pytest
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.scorer import SufficiencyScorer


@pytest.fixture
def scorer():
    return SufficiencyScorer()


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence)


def all_inactive() -> list[DetectorResult]:
    return [make_result(dim) for dim in Dimension]


class TestEmotionGate:
    """Emotion is the gatekeeper — without it, ring can't pass 45%."""

    def test_no_emotion_caps_at_45(self, scorer):
        """Even if other dimensions fire, no emotion = capped."""
        results = all_inactive()
        # Activate 5 non-emotion dimensions
        for dim in [Dimension.CONFLICT, Dimension.HUMOR, Dimension.MBTI, Dimension.EQ, Dimension.FRAGILITY]:
            results = [r if r.dimension != dim else make_result(dim, True, 0.8) for r in results]
        report = scorer.score(results)
        assert report.score <= 0.45

    def test_emotion_alone_gives_progress(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION else make_result(Dimension.EMOTION, True, 0.6)
            for r in results
        ]
        report = scorer.score(results)
        assert 0.1 < report.score < 0.5

    def test_emotion_unlocks_higher_scores(self, scorer):
        results = all_inactive()
        for dim in [Dimension.EMOTION, Dimension.CONFLICT, Dimension.FRAGILITY, Dimension.EQ]:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
        report = scorer.score(results)
        assert report.score > 0.45


class TestActivationThreshold:
    """7 of 11 activated = 100%."""

    def test_seven_activated_is_full(self, scorer):
        dims = [Dimension.EMOTION, Dimension.CONFLICT, Dimension.HUMOR, Dimension.MBTI,
                Dimension.EQ, Dimension.FRAGILITY, Dimension.COMMUNICATION_DNA]
        results = all_inactive()
        for dim in dims:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
        report = scorer.score(results)
        assert report.score == 1.0
        assert report.ready is True

    def test_six_activated_not_full(self, scorer):
        dims = [Dimension.EMOTION, Dimension.CONFLICT, Dimension.HUMOR, Dimension.MBTI,
                Dimension.EQ, Dimension.FRAGILITY]
        results = all_inactive()
        for dim in dims:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
        report = scorer.score(results)
        assert report.score < 1.0
        assert report.ready is False

    def test_seven_without_emotion_still_capped(self, scorer):
        """7 activated but no emotion → still capped."""
        dims = [Dimension.CONFLICT, Dimension.HUMOR, Dimension.MBTI, Dimension.EQ,
                Dimension.FRAGILITY, Dimension.COMMUNICATION_DNA, Dimension.SOULGRAPH]
        results = all_inactive()
        for dim in dims:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
        report = scorer.score(results)
        assert report.score <= 0.45


class TestProgressGradient:
    """Score increases smoothly as more dimensions activate."""

    def test_zero_input_zero_score(self, scorer):
        report = scorer.score(all_inactive())
        assert report.score == 0.0
        assert report.activated_count == 0

    def test_monotonic_increase(self, scorer):
        dims_ordered = [
            Dimension.EMOTION, Dimension.EQ, Dimension.CONFLICT,
            Dimension.FRAGILITY, Dimension.HUMOR, Dimension.MBTI,
            Dimension.COMMUNICATION_DNA,
        ]
        results = all_inactive()
        prev_score = 0.0
        for dim in dims_ordered:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
            report = scorer.score(results)
            assert report.score >= prev_score, f"Score decreased when adding {dim}"
            prev_score = report.score


class TestRingSegments:
    def test_segments_count(self, scorer):
        report = scorer.score(all_inactive())
        assert len(report.segments) == 11

    def test_activated_segment_is_filled(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION else make_result(Dimension.EMOTION, True, 0.8)
            for r in results
        ]
        report = scorer.score(results)
        emotion_seg = [s for s in report.segments if s.dimension == Dimension.EMOTION][0]
        assert emotion_seg.filled is True
        assert emotion_seg.intensity == 0.8


class TestPromptHint:
    def test_no_activation_tells_more(self, scorer):
        report = scorer.score(all_inactive())
        assert report.prompt_hint == "tell_me_more"

    def test_ready_when_full(self, scorer):
        dims = [Dimension.EMOTION, Dimension.CONFLICT, Dimension.HUMOR, Dimension.MBTI,
                Dimension.EQ, Dimension.FRAGILITY, Dimension.COMMUNICATION_DNA]
        results = all_inactive()
        for dim in dims:
            results = [r if r.dimension != dim else make_result(dim, True, 0.7) for r in results]
        report = scorer.score(results)
        assert report.prompt_hint == "ready"

    def test_no_emotion_asks_feelings(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EQ else make_result(Dimension.EQ, True, 0.5)
            for r in results
        ]
        report = scorer.score(results)
        assert report.prompt_hint == "how_do_you_feel"
