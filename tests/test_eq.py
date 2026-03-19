"""Tests for the EQ (behavioral features) adapter — zero LLM cost."""

import pytest
from sufficiency_scorer.detectors.eq import extract_behavioral, compute_valence, compute_distress, EQAdapter


class TestBehavioralExtraction:
    def test_empty_text(self):
        features = extract_behavioral("")
        assert features["words"] == 0
        assert features["self_ref"] == 0.0

    def test_self_reference(self):
        features = extract_behavioral("I feel like I'm losing my mind and I can't stop it")
        assert features["self_ref"] > 0.1
        assert features["words"] > 5

    def test_question_detection(self):
        features = extract_behavioral("What am I supposed to do? Why does this happen?")
        assert features["question_ratio"] > 0.0

    def test_negative_emotion_words(self):
        features = extract_behavioral("I'm so stressed and anxious about everything")
        assert features["neg_emotion_ratio"] > 0.0

    def test_positive_emotion_words(self):
        features = extract_behavioral("I'm so happy and excited about this wonderful news")
        assert features["pos_emotion_ratio"] > 0.0

    def test_hedging(self):
        features = extract_behavioral("Maybe I should probably think about it")
        assert features["hedging_ratio"] > 0.0

    def test_absolutist(self):
        features = extract_behavioral("Nothing ever works and everything is always broken")
        assert features["absolutist_ratio"] > 0.0

    def test_gibberish_low_signal(self):
        features = extract_behavioral("haha hi lol ok")
        assert features["self_ref"] == 0.0
        assert features["neg_emotion_ratio"] == 0.0


class TestValenceDistress:
    def test_negative_valence(self):
        features = extract_behavioral("I'm stressed and frustrated with everything")
        v = compute_valence(features)
        assert v < 0

    def test_positive_valence(self):
        features = extract_behavioral("I'm so happy and excited about life")
        v = compute_valence(features)
        assert v > 0

    def test_distress_high_for_negative(self):
        features = extract_behavioral("I'm so stressed and anxious I can't sleep")
        d = compute_distress(features)
        assert d > 0.1

    def test_distress_low_for_neutral(self):
        features = extract_behavioral("the weather is nice today")
        d = compute_distress(features)
        assert d < 0.2


class TestEQAdapter:
    @pytest.mark.asyncio
    async def test_gibberish_not_activated(self):
        adapter = EQAdapter()
        result = await adapter.detect("haha hi")
        assert result.activated is False or result.confidence < 0.15

    @pytest.mark.asyncio
    async def test_emotional_text_activates(self):
        adapter = EQAdapter()
        result = await adapter.detect(
            "I'm really stressed and anxious about my job. I don't know what to do anymore."
        )
        assert result.activated is True
        assert result.confidence > 0.2

    @pytest.mark.asyncio
    async def test_short_text_not_activated(self):
        adapter = EQAdapter()
        result = await adapter.detect("ok")
        assert result.activated is False
