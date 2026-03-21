"""Tests for V8x star generation logic."""

import pytest
from sufficiency_scorer.v8x_star_generator import StarGenerator


@pytest.fixture
def gen():
    return StarGenerator()


class TestShouldGenerateStar:
    """Tests for should_generate_star."""

    def test_new_dimension_generates_star(self, gen):
        """New 'emotion' with confidence 0.8 -> generate=True."""
        result = gen.should_generate_star(
            detector_result={"dimension": "emotion", "confidence": 0.8, "detail": {"frustration": 0.7}},
            existing_stars=[],
            turn_number=1,
        )
        assert result["generate"] is True
        assert result["dimension"] == "emotion"
        assert result["star_type"] in ("bright", "dim")

    def test_existing_dimension_no_new_star(self, gen):
        """'emotion' already in stars -> generate=False."""
        result = gen.should_generate_star(
            detector_result={"dimension": "emotion", "confidence": 0.8, "detail": {"frustration": 0.7}},
            existing_stars=["emotion"],
            turn_number=2,
        )
        assert result["generate"] is False

    def test_low_confidence_no_star(self, gen):
        """Confidence <= 0.5 should not generate."""
        result = gen.should_generate_star(
            detector_result={"dimension": "emotion", "confidence": 0.4, "detail": {}},
            existing_stars=[],
            turn_number=1,
        )
        assert result["generate"] is False

    def test_max_one_per_turn(self, gen):
        """3 new dimensions -> only 1 star (highest confidence)."""
        results = [
            {"dimension": "emotion", "confidence": 0.6, "detail": {"frustration": 0.7}},
            {"dimension": "conflict", "confidence": 0.9, "detail": {"style": "avoid"}},
            {"dimension": "humor", "confidence": 0.7, "detail": {"style": "affiliative"}},
        ]
        stars = gen.generate_stars_for_turn(
            detector_results=results,
            existing_stars=[],
            turn_number=1,
        )
        # Should only produce 1 star from detection (highest confidence = conflict at 0.9)
        assert len(stars) == 1
        assert stars[0]["dimension"] == "conflict"


class TestMinimumGuarantee:
    """Tests for enforce_minimum_guarantee."""

    def test_minimum_guarantee_turn_2(self, gen):
        """Only 1 star at turn 2 -> force-generate 1 more."""
        forced = gen.enforce_minimum_guarantee(
            existing_stars=["emotion"],
            turn_number=2,
            detector_results=[
                {"dimension": "conflict", "confidence": 0.3, "detail": {"style": "avoid"}},
            ],
            user_topics=["relationships"],
        )
        assert len(forced) >= 1

    def test_minimum_guarantee_turn_4(self, gen):
        """Only 2 stars at turn 4 -> force-generate 1 more."""
        forced = gen.enforce_minimum_guarantee(
            existing_stars=["emotion", "conflict"],
            turn_number=4,
            detector_results=[],
            user_topics=["career", "stress"],
        )
        assert len(forced) >= 1

    def test_no_force_when_minimum_met(self, gen):
        """3 stars at turn 4 -> no force."""
        forced = gen.enforce_minimum_guarantee(
            existing_stars=["emotion", "conflict", "humor"],
            turn_number=4,
            detector_results=[],
            user_topics=[],
        )
        assert len(forced) == 0

    def test_fallback_uses_topics(self, gen):
        """No new detections but need more stars -> use user_topics."""
        forced = gen.enforce_minimum_guarantee(
            existing_stars=["emotion"],
            turn_number=2,
            detector_results=[],
            user_topics=["work stress"],
        )
        assert len(forced) >= 1
        # The forced star should exist
        assert forced[0]["dimension"] is not None


class TestGenerateStarsIntegration:
    """Tests for generate_stars_for_turn (full flow)."""

    def test_generate_stars_integration(self, gen):
        """Full flow with multiple detectors."""
        results = [
            {"dimension": "emotion", "confidence": 0.8, "detail": {"frustration": 0.7}},
            {"dimension": "conflict", "confidence": 0.7, "detail": {"style": "avoid"}},
        ]
        stars = gen.generate_stars_for_turn(
            detector_results=results,
            existing_stars=[],
            turn_number=2,
            user_topics=["family"],
        )
        # Should have at least 2 stars (minimum guarantee for turn 2)
        assert len(stars) >= 2
        # Each star should have required keys
        for star in stars:
            assert "dimension" in star
            assert "label" in star
            assert "star_type" in star
            assert "star_color" in star

    def test_integration_respects_existing(self, gen):
        """Should not duplicate existing dimensions."""
        results = [
            {"dimension": "humor", "confidence": 0.9, "detail": {"style": "affiliative"}},
        ]
        stars = gen.generate_stars_for_turn(
            detector_results=results,
            existing_stars=["emotion", "conflict"],
            turn_number=4,
            user_topics=["music"],
        )
        dims = [s["dimension"] for s in stars]
        assert "emotion" not in dims
        assert "conflict" not in dims
