"""Tests for V8x fog signal emitter (FR-08)."""

import pytest
from sufficiency_scorer.v8x_fog_signals import FogSignalEmitter


@pytest.fixture
def emitter():
    return FogSignalEmitter()


class TestFogDisturbanceFormat:
    def test_fog_disturbance_format(self, emitter):
        signal = emitter.create_fog_disturbance("emotion", 0.8, [])
        assert signal["event"] == "fog_disturbance"
        assert signal["dimension"] == "emotion"
        assert "x" in signal["position"]
        assert "y" in signal["position"]
        assert signal["intensity"] == 0.8
        assert signal["will_become_star"] is True
        assert signal["eta_ms"] == 3000
        assert isinstance(signal["timestamp_ms"], int)


class TestStarCreatedFormat:
    def test_star_created_format(self, emitter):
        pos = {"x": 0.5, "y": 0.5}
        signal = emitter.create_star_created("emotion", "Warm heart", "rose", pos)
        assert signal["event"] == "star_created"
        assert signal["dimension"] == "emotion"
        assert signal["label"] == "Warm heart"
        assert signal["star_color"] == "rose"
        assert signal["position"] == pos
        assert isinstance(signal["timestamp_ms"], int)


class TestPositionCalculation:
    def test_position_first_star_center(self, emitter):
        pos = emitter.calculate_position("emotion", [])
        assert pos["x"] == 0.5
        assert pos["y"] == 0.5

    def test_position_avoids_overlap(self, emitter):
        existing = [{"dimension": "humor", "position": {"x": 0.5, "y": 0.5}}]
        pos = emitter.calculate_position("conflict", existing)
        # Must not be at center
        dx = pos["x"] - 0.5
        dy = pos["y"] - 0.5
        dist = (dx**2 + dy**2) ** 0.5
        assert dist >= 0.15, f"New star too close to existing: distance={dist}"


class TestSignalsForTurn:
    def test_signals_for_turn_order(self, emitter):
        new_stars = [
            {"dimension": "emotion", "confidence": 0.9, "label": "Warm", "star_color": "rose"},
        ]
        signals = emitter.signals_for_turn(new_stars, [])
        events = [s["event"] for s in signals]
        assert events == ["fog_disturbance", "star_created"]

    def test_signals_for_turn_count(self, emitter):
        new_stars = [
            {"dimension": "emotion", "confidence": 0.9, "label": "Warm", "star_color": "rose"},
            {"dimension": "humor", "confidence": 0.7, "label": "Witty", "star_color": "gold"},
        ]
        signals = emitter.signals_for_turn(new_stars, [])
        assert len(signals) == 4
        fog_signals = [s for s in signals if s["event"] == "fog_disturbance"]
        star_signals = [s for s in signals if s["event"] == "star_created"]
        assert len(fog_signals) == 2
        assert len(star_signals) == 2


class TestIntensityAndEta:
    def test_intensity_matches_confidence(self, emitter):
        signal = emitter.create_fog_disturbance("emotion", 0.65, [])
        assert signal["intensity"] == 0.65

    def test_eta_is_3000ms(self, emitter):
        signal = emitter.create_fog_disturbance("emotion", 0.5, [])
        assert signal["eta_ms"] == 3000
