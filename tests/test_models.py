"""Tests for insight-oriented data models."""

import pytest
from sufficiency_scorer.models import (
    Dimension,
    DetectorResult,
    InsightCandidate,
    InsightQuality,
    SufficiencyReport,
    SessionState,
)


class TestInsightCandidate:
    def test_create_valid_insight(self):
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
            signal="high frustration + avoidance style",
            reframe="You pick your battles carefully — not out of weakness, but because you know which fights matter",
            quality=InsightQuality.HIGH,
            confidence=0.75,
        )
        assert insight.quality == InsightQuality.HIGH
        assert len(insight.source_dimensions) == 2

    def test_insight_quality_ordering(self):
        assert InsightQuality.HIGH.value > InsightQuality.MEDIUM.value
        assert InsightQuality.MEDIUM.value > InsightQuality.LOW.value
        assert InsightQuality.LOW.value > InsightQuality.NOISE.value


class TestSufficiencyReport:
    def test_ready_with_3_high_quality(self):
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="strong frustration",
                reframe="You have a sharp sense of fairness",
                quality=InsightQuality.HIGH,
                confidence=0.8,
            )
            for _ in range(3)
        ]
        report = SufficiencyReport(
            ready=True,
            insights=insights,
            detector_results=[],
            ring_progress=1.0,
            prompt_hint="ready",
        )
        assert report.ready is True
        assert len(report.insights) == 3

    def test_not_ready_with_2_insights(self):
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="moderate sadness",
                reframe="You feel things deeply",
                quality=InsightQuality.HIGH,
                confidence=0.7,
            )
            for _ in range(2)
        ]
        report = SufficiencyReport(
            ready=False,
            insights=insights,
            detector_results=[],
            ring_progress=0.67,
            prompt_hint="keep_going",
        )
        assert report.ready is False

    def test_ring_progress_clamped(self):
        report = SufficiencyReport(
            ready=True, insights=[], detector_results=[],
            ring_progress=1.0, prompt_hint="ready",
        )
        assert 0.0 <= report.ring_progress <= 1.0


class TestSessionState:
    def test_accumulate_text(self):
        state = SessionState()
        state.add_segment("I'm really stressed about work.")
        assert state.word_count >= 5
        assert len(state.segments) == 1

    def test_multi_segment_accumulation(self):
        state = SessionState()
        state.add_segment("My boss keeps pushing me.")
        state.add_segment("I don't know what to do anymore.")
        assert len(state.segments) == 2
        assert "boss" in state.full_text
        assert "anymore" in state.full_text

    def test_word_count_tracks_total(self):
        state = SessionState()
        state.add_segment("one two three four five")
        state.add_segment("six seven eight nine ten")
        assert state.word_count == 10

    def test_meets_minimum_words(self):
        state = SessionState(min_words=40)
        state.add_segment("Short text here.")
        assert state.meets_minimum is False
        state.add_segment(
            "I have been feeling really overwhelmed lately with everything "
            "going on at work and my boss keeps asking me to do overtime "
            "and I just cannot take it anymore honestly it is getting to be "
            "way too much for me to handle on my own"
        )
        assert state.meets_minimum is True
