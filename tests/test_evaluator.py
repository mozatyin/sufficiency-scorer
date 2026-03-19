"""Tests for the insight quality evaluation framework."""

import pytest
from sufficiency_scorer.evaluator import InsightEvaluator, EvalResult
from sufficiency_scorer.models import InsightCandidate, InsightQuality, Dimension


class TestSpecificityScore:
    """Insights must be specific to the user, not generic platitudes."""

    def test_generic_platitude_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="frustration: 0.55",
            reframe="You're a good person",
            quality=InsightQuality.MEDIUM,
            confidence=0.7,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity < 0.5

    def test_contextualized_insight_scores_high(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
            signal="frustration (55%) + avoidance (70%)",
            reframe="You feel the friction but choose restraint — especially when your boss is forcing overtime",
            quality=InsightQuality.HIGH,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity >= 0.5

    def test_mere_repetition_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="frustration: 0.55",
            reframe="You feel frustrated",
            quality=InsightQuality.MEDIUM,
            confidence=0.7,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity < 0.3


class TestReframeScore:
    """Reframes must transform, not repeat."""

    def test_positive_reframe_scores_well(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.FRAGILITY],
            signal="fragility pattern: open",
            reframe="You let yourself be seen — that takes more courage than most realize",
            quality=InsightQuality.MEDIUM,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.reframe_quality >= 0.5

    def test_negative_framing_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="anxiety: 0.6",
            reframe="You're very anxious and worried about everything",
            quality=InsightQuality.MEDIUM,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.reframe_quality < 0.5


class TestBatchEval:
    """Evaluate a full set of insights."""

    def test_batch_returns_summary(self):
        evaluator = InsightEvaluator()
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="frustration: 0.55",
                reframe="You have a sharp sense of when things aren't right — in your work situation",
                quality=InsightQuality.MEDIUM,
                confidence=0.7,
            ),
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
                signal="frustration + avoidance",
                reframe="You feel the friction but choose restraint — you're more strategic than you give yourself credit for",
                quality=InsightQuality.HIGH,
                confidence=0.6,
            ),
            InsightCandidate(
                source_dimensions=[Dimension.FRAGILITY],
                signal="open fragility",
                reframe="You let yourself be seen — that takes more courage than most realize",
                quality=InsightQuality.MEDIUM,
                confidence=0.5,
            ),
        ]
        summary = evaluator.evaluate_batch(insights)
        assert summary.total == 3
        assert 0.0 <= summary.avg_specificity <= 1.0
        assert 0.0 <= summary.avg_reframe_quality <= 1.0
        assert summary.bloom_worthy is not None  # bool

    def test_empty_batch(self):
        evaluator = InsightEvaluator()
        summary = evaluator.evaluate_batch([])
        assert summary.total == 0
        assert summary.bloom_worthy is False
