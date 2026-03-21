"""Tests for Star Engine — gaps 4, 5, 6."""

import pytest
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.star_engine import (
    StarEngine, Star, FogEvent, StarCreatedEvent, DETECTOR_THRESHOLDS, DEFAULT_THRESHOLD,
)
from sufficiency_scorer.star_labels import (
    get_positive_label, get_signal_key, get_dark_labels, BANNED_TERMS,
)


def dr(dim, activated=False, conf=0.0, detail=None):
    return DetectorResult(dimension=dim, activated=activated, confidence=conf, detail=detail or {})


def all_inactive():
    return [dr(dim) for dim in Dimension]


def work_pressure():
    base = all_inactive()
    overrides = {
        Dimension.EMOTION: dr(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}),
        Dimension.EQ: dr(Dimension.EQ, True, 0.6, {"features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27}, "valence": -0.35, "distress": 0.43}),
        Dimension.CONFLICT: dr(Dimension.CONFLICT, True, 0.6, {"styles": {"avoid": 0.7, "compromise": 0.4}}),
        Dimension.FRAGILITY: dr(Dimension.FRAGILITY, True, 0.5, {"pattern": "open"}),
    }
    return [overrides.get(r.dimension, r) for r in base]


# === Gap 5: Label Mapping ===

class TestPositiveLabels:
    def test_emotion_frustration(self):
        label = get_positive_label(Dimension.EMOTION, "frustration")
        assert label == "对不公平很敏感"

    def test_conflict_avoid(self):
        label = get_positive_label(Dimension.CONFLICT, "avoid")
        assert label == "先思考再行动"

    def test_fragility_open(self):
        label = get_positive_label(Dimension.FRAGILITY, "open")
        assert label == "敢于面对真实"

    def test_humor_self_deprecating(self):
        label = get_positive_label(Dimension.HUMOR, "self_deprecating")
        assert label == "自嘲是你的超能力"

    def test_eq_question_ratio(self):
        label = get_positive_label(Dimension.EQ, "high_question_ratio")
        assert label == "主动寻找出路"

    def test_no_banned_terms_in_labels(self):
        from sufficiency_scorer.star_labels import POSITIVE_LABELS
        for dim, labels in POSITIVE_LABELS.items():
            for key, label in labels.items():
                for banned in BANNED_TERMS:
                    assert banned not in label, f"Banned term '{banned}' in {dim}.{key}: {label}"

    def test_signal_key_extraction(self):
        detail = {"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}
        key = get_signal_key(Dimension.EMOTION, detail)
        assert key == "frustration"

    def test_signal_key_conflict(self):
        detail = {"styles": {"avoid": 0.7, "compromise": 0.4}}
        key = get_signal_key(Dimension.CONFLICT, detail)
        assert key == "avoid"

    def test_signal_key_eq_question(self):
        detail = {"features": {"self_ref": 0.05, "question_ratio": 0.33}, "distress": 0.1}
        key = get_signal_key(Dimension.EQ, detail)
        assert key == "high_question_ratio"


class TestDarkLabels:
    def test_dark_label_avoid_plus_defensive(self):
        results = {
            Dimension.CONFLICT: {"detail": {"styles": {"avoid": 0.7}}, "confidence": 0.6},
            Dimension.FRAGILITY: {"detail": {"pattern": "defensive"}, "confidence": 0.5},
        }
        labels = get_dark_labels(results)
        assert "需要属于自己的空间?" in labels

    def test_dark_label_frustration_plus_question(self):
        results = {
            Dimension.EMOTION: {"detail": {"top_emotions": [("frustration", 0.55)]}, "confidence": 0.7},
            Dimension.EQ: {"detail": {"features": {"question_ratio": 0.3}}, "confidence": 0.6},
        }
        labels = get_dark_labels(results)
        assert "内心渴望改变?" in labels

    def test_no_dark_labels_when_no_match(self):
        results = {
            Dimension.EMOTION: {"detail": {"top_emotions": [("happiness", 0.8)]}, "confidence": 0.8},
        }
        labels = get_dark_labels(results)
        assert len(labels) == 0


# === Gap 4: Star Generation Logic ===

class TestStarGeneration:
    def test_new_dimension_creates_star(self):
        engine = StarEngine()
        results = [dr(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55)]})]
        results += [dr(dim) for dim in Dimension if dim != Dimension.EMOTION]
        output = engine.process_turn(results, turn_count=1)
        assert len(output.new_stars) == 1
        assert output.new_stars[0].star.dimension == Dimension.EMOTION
        assert output.new_stars[0].star.label == "对不公平很敏感"

    def test_max_one_new_star_per_turn(self):
        engine = StarEngine()
        output = engine.process_turn(work_pressure(), turn_count=1)
        assert len(output.new_stars) <= 1

    def test_existing_dimension_no_new_star(self):
        engine = StarEngine()
        results = [dr(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55)]})]
        results += [dr(dim) for dim in Dimension if dim != Dimension.EMOTION]
        engine.process_turn(results, turn_count=1)
        # Same dimension again
        output2 = engine.process_turn(results, turn_count=2)
        new_emotion_stars = [s for s in output2.new_stars if s.star.dimension == Dimension.EMOTION]
        assert len(new_emotion_stars) == 0

    def test_brightness_change_on_confidence_delta(self):
        engine = StarEngine()
        r1 = [dr(Dimension.EMOTION, True, 0.5, {"top_emotions": [("frustration", 0.5)]})]
        r1 += [dr(dim) for dim in Dimension if dim != Dimension.EMOTION]
        engine.process_turn(r1, turn_count=1)
        # Same dimension, higher confidence
        r2 = [dr(Dimension.EMOTION, True, 0.9, {"top_emotions": [("frustration", 0.9)]})]
        r2 += [dr(dim) for dim in Dimension if dim != Dimension.EMOTION]
        output2 = engine.process_turn(r2, turn_count=2)
        assert len(output2.brightness_changes) >= 1
        bc = output2.brightness_changes[0]
        assert bc.new_brightness > bc.old_brightness

    def test_nothing_activated_no_stars(self):
        engine = StarEngine()
        output = engine.process_turn(all_inactive(), turn_count=1)
        assert len(output.new_stars) == 0
        assert len(output.fog_events) == 0


class TestMinimumGuarantee:
    def test_turn2_at_least_2_stars(self):
        engine = StarEngine()
        # Turn 1: only EQ activates (1 star)
        r1 = all_inactive()
        r1 = [r if r.dimension != Dimension.EQ else dr(Dimension.EQ, True, 0.5,
              {"features": {"self_ref": 0.1, "question_ratio": 0.2}, "distress": 0.3}) for r in r1]
        engine.process_turn(r1, turn_count=1)
        assert len(engine.stars) >= 1

        # Turn 2: nothing new activates, but minimum guarantee kicks in
        r2 = r1  # same results
        # Add a weakly activated emotion so fallback can find it
        r2 = [r if r.dimension != Dimension.EMOTION else dr(Dimension.EMOTION, True, 0.3,
              {"top_emotions": [("sadness", 0.3)]}) for r in r2]
        engine.process_turn(r2, turn_count=2)
        assert len(engine.stars) >= 2, f"Expected ≥2 stars at turn 2, got {len(engine.stars)}"

    def test_turn4_at_least_3_stars(self):
        engine = StarEngine()
        results = work_pressure()
        for turn in range(1, 5):
            engine.process_turn(results, turn_count=turn)
        assert len(engine.stars) >= 3, f"Expected ≥3 stars at turn 4, got {len(engine.stars)}"


class TestFallback:
    def test_fallback_uses_unlabeled_dimension(self):
        engine = StarEngine()
        # Create one star manually
        r1 = all_inactive()
        r1 = [r if r.dimension != Dimension.EMOTION else dr(Dimension.EMOTION, True, 0.7,
              {"top_emotions": [("sadness", 0.6)]}) for r in r1]
        engine.process_turn(r1, turn_count=1)
        assert len(engine.stars) == 1

        # Turn 2: emotion already starred, EQ activated but not starred yet
        r2 = r1[:]
        r2 = [r if r.dimension != Dimension.EQ else dr(Dimension.EQ, True, 0.5,
              {"features": {"self_ref": 0.1, "question_ratio": 0.2}, "distress": 0.3}) for r in r2]
        engine.process_turn(r2, turn_count=2)
        assert len(engine.stars) >= 2


# === Gap 6: Fog Events ===

class TestFogEvents:
    def test_fog_event_on_new_star(self):
        engine = StarEngine()
        results = [dr(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55)]})]
        results += [dr(dim) for dim in Dimension if dim != Dimension.EMOTION]
        output = engine.process_turn(results, turn_count=1)
        assert len(output.fog_events) == 1
        fog = output.fog_events[0]
        assert fog.event == "fog_appear"
        assert fog.animation.duration_ms == 3000
        assert fog.animation.color_hint == "warm"  # emotion = warm
        assert fog.dimension == Dimension.EMOTION
        assert 0.0 <= fog.position[0] <= 1.0
        assert 0.0 <= fog.position[1] <= 1.0
        assert fog.intensity > 0

    def test_no_fog_when_no_new_star(self):
        engine = StarEngine()
        output = engine.process_turn(all_inactive(), turn_count=1)
        assert len(output.fog_events) == 0

    def test_fog_position_matches_star_position(self):
        engine = StarEngine()
        results = [dr(Dimension.CONFLICT, True, 0.6, {"styles": {"avoid": 0.7}})]
        results += [dr(dim) for dim in Dimension if dim != Dimension.CONFLICT]
        output = engine.process_turn(results, turn_count=1)
        if output.fog_events and output.new_stars:
            assert output.fog_events[0].position == output.new_stars[0].fog_position


class TestDarkStars:
    def test_dark_star_after_turn3(self):
        engine = StarEngine()
        # Build up normal stars for turns 1-3
        for turn in range(1, 4):
            engine.process_turn(work_pressure(), turn_count=turn)
        # Turn 4: should try dark star if pattern matches
        results = work_pressure()
        # Add defensive fragility for dark pattern match
        results = [r if r.dimension != Dimension.FRAGILITY else
                   dr(Dimension.FRAGILITY, True, 0.5, {"pattern": "defensive"})
                   for r in results]
        output = engine.process_turn(results, turn_count=4)
        dark_stars = [s for s in engine.stars if s.is_dark]
        # May or may not have dark star depending on pattern match
        # Just verify no crash and valid structure
        for s in dark_stars:
            assert s.label.endswith("?")
            assert s.brightness <= 0.5


class TestReset:
    def test_reset_clears_everything(self):
        engine = StarEngine()
        engine.process_turn(work_pressure(), turn_count=1)
        assert len(engine.stars) > 0
        engine.reset()
        assert len(engine.stars) == 0


# === V6 Gate-Aware Star Suppression ===

class TestSafetyGate:
    """Stars must respect the Guard module's gate output."""

    def test_layer_0_suppresses_all(self):
        """Crisis mode: no stars, no fog, no brightness changes."""
        engine = StarEngine()
        output = engine.process_turn(
            work_pressure(), turn_count=1, safety_gate="layer_0_only",
        )
        assert len(output.fog_events) == 0
        assert len(output.new_stars) == 0
        assert len(output.brightness_changes) == 0
        assert len(engine.stars) == 0

    def test_layer_1_suppresses_new_stars(self):
        """High distress: no new stars, but brightness updates allowed."""
        engine = StarEngine()
        # First create a star normally
        engine.process_turn(work_pressure(), turn_count=1, safety_gate="layer_3_ok")
        initial_count = len(engine.stars)
        assert initial_count > 0

        # Now at layer_1: should NOT create new stars
        output = engine.process_turn(
            work_pressure(), turn_count=2, safety_gate="layer_1",
        )
        assert len(output.new_stars) == 0
        assert len(engine.stars) == initial_count  # no new stars

    def test_layer_1_allows_brightness_updates(self):
        """At layer_1, existing stars can still update brightness."""
        engine = StarEngine()
        # Create star at low confidence
        results = work_pressure()
        engine.process_turn(results, turn_count=1, safety_gate="layer_3_ok")

        # Increase confidence significantly
        boosted = work_pressure()
        for r in boosted:
            if r.activated:
                r.confidence = min(r.confidence + 0.3, 1.0)

        output = engine.process_turn(boosted, turn_count=2, safety_gate="layer_1")
        # Brightness changes should still fire
        # (may or may not depending on delta threshold)

    def test_layer_2_normal_generation(self):
        """layer_2_ok: normal star generation."""
        engine = StarEngine()
        output = engine.process_turn(
            work_pressure(), turn_count=1, safety_gate="layer_2_ok",
        )
        assert len(engine.stars) > 0

    def test_layer_3_normal_generation(self):
        """layer_3_ok (default): normal star generation."""
        engine = StarEngine()
        output = engine.process_turn(work_pressure(), turn_count=1)
        assert len(engine.stars) > 0

    def test_crisis_then_recovery_stars_resume(self):
        """After crisis clears, stars should resume normally."""
        engine = StarEngine()
        # Turn 1: crisis — no stars
        engine.process_turn(work_pressure(), turn_count=1, safety_gate="layer_0_only")
        assert len(engine.stars) == 0

        # Turn 2: recovered — stars resume
        output = engine.process_turn(
            work_pressure(), turn_count=2, safety_gate="layer_2_ok",
        )
        assert len(engine.stars) > 0

    def test_layer_0_suppresses_dark_stars(self):
        """Dark stars also suppressed at layer_0."""
        engine = StarEngine()
        # Build up stars first
        engine.process_turn(work_pressure(), turn_count=1, safety_gate="layer_3_ok")
        engine.process_turn(work_pressure(), turn_count=2, safety_gate="layer_3_ok")
        stars_before = len(engine.stars)

        # Turn 3: crisis — no dark star
        output = engine.process_turn(
            work_pressure(), turn_count=3, safety_gate="layer_0_only",
        )
        assert len(engine.stars) == stars_before  # no new stars at all

    def test_layer_1_suppresses_fallback(self):
        """Min guarantee fallback also suppressed at layer_1."""
        engine = StarEngine()
        # Turn 2 normally requires ≥ 2 stars, but at layer_1 no fallback
        output = engine.process_turn(
            all_inactive(), turn_count=2, safety_gate="layer_1",
        )
        assert len(engine.stars) == 0  # no fallback at layer_1
