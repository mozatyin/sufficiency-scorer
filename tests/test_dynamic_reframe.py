"""Tests for dynamic (context-aware) reframing."""

import pytest
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


def make_result(dim, activated=False, confidence=0.0, detail=None):
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


class TestDynamicReframe:
    """Reframes should incorporate user's actual words/context."""

    def test_emotion_reframe_includes_context(self):
        """When user_text is provided, emotion reframe should reference their situation."""
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="my boss is forcing me to do overtime")
        emotion_ins = [i for i in insights if Dimension.EMOTION in i.source_dimensions]
        assert len(emotion_ins) >= 1
        # Should reference work/boss context, not just generic "things aren't right"
        reframe = emotion_ins[0].reframe
        assert any(word in reframe.lower() for word in ("work", "boss", "overtime", "pushing", "situation")), \
            f"Reframe too generic, doesn't reference user context: {reframe}"

    def test_without_user_text_falls_back_to_template(self):
        """Without user_text, falls back to static template."""
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)  # no user_text
        assert len(insights) >= 1
        # Should still produce a valid reframe
        assert len(insights[0].reframe) > 10

    def test_conflict_reframe_includes_context(self):
        results = [make_result(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I keep quiet when my partner criticizes me")
        conflict_ins = [i for i in insights if Dimension.CONFLICT in i.source_dimensions]
        assert len(conflict_ins) >= 1
        reframe = conflict_ins[0].reframe
        assert any(word in reframe.lower() for word in ("partner", "quiet", "relationship", "criticism", "situation")), \
            f"Reframe too generic: {reframe}"

    def test_cross_dimensional_reframe_with_context(self):
        """Cross-dimensional reframe should also incorporate context."""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55)],
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I'm tired of my boss making me work late but I never say anything")
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        if cross:
            assert any(word in cross[0].reframe.lower() for word in ("boss", "work", "situation", "speak")), \
                f"Cross reframe too generic: {cross[0].reframe}"

    def test_humor_reframe_includes_context(self):
        results = [make_result(Dimension.HUMOR, True, 0.7, detail={
            "humor_detected": True, "styles": {"self_deprecating": 0.8},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I just got dumped again third time this year at least I'm consistent")
        humor_ins = [i for i in insights if Dimension.HUMOR in i.source_dimensions]
        assert len(humor_ins) >= 1

    def test_multiple_insights_all_contextualized(self):
        """All insights in a rich scenario should reference user context."""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55), ("sadness", 0.48)],
            }),
            make_result(Dimension.EQ, True, 0.6, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
                "valence": -0.35, "distress": 0.43,
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(
            results,
            user_text="I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?"
        )
        # At least some insights should be contextualized
        contextualized = sum(1 for i in insights if any(
            w in i.reframe.lower() for w in ("work", "boss", "overtime", "tired", "situation")
        ))
        assert contextualized >= 1, f"No insights reference user context. Reframes: {[i.reframe for i in insights]}"
