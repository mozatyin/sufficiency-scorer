"""Evaluate insight quality across all 3 scenarios.

Runs InsightExtractor with user_text (dynamic reframing) then
InsightEvaluator to verify bloom-worthiness across scenario types.
"""

import pytest
from sufficiency_scorer.evaluator import InsightEvaluator
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


def make_result(dim, activated=False, confidence=0.0, detail=None):
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


SCENARIO_CONFIGS = {
    "工作压力": {
        "text": "hey, I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}),
            make_result(Dimension.EQ, True, 0.6, {"features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27}, "valence": -0.35, "distress": 0.43}),
            make_result(Dimension.CONFLICT, True, 0.6, {"styles": {"avoid": 0.7, "compromise": 0.4}}),
            make_result(Dimension.FRAGILITY, True, 0.5, {"pattern": "open", "pattern_scores": {"open": 0.6}}),
        ],
    },
    "自嘲幽默": {
        "text": "haha so I just got dumped again, third time this year. I'm starting to think maybe I'm the common denominator here. At least I'm consistent right?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.6, {"top_emotions": [("sadness", 0.5), ("amusement", 0.4)]}),
            make_result(Dimension.HUMOR, True, 0.7, {"humor_detected": True, "styles": {"self_deprecating": 0.8}}),
            make_result(Dimension.FRAGILITY, True, 0.5, {"pattern": "masked", "pattern_scores": {"masked": 0.6}}),
            make_result(Dimension.EQ, True, 0.5, {"features": {"self_ref": 0.1, "question_ratio": 0.33, "words": 26}, "valence": -0.15, "distress": 0.25}),
            make_result(Dimension.CONFLICT, True, 0.5, {"styles": {"avoid": 0.6, "accommodating": 0.3}}),
        ],
    },
    "丧亲之痛": {
        "text": "my mom passed away last month and I don't know how to be normal anymore. People keep saying it gets better but when?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.8, {"top_emotions": [("grief", 0.8), ("sadness", 0.7)]}),
            make_result(Dimension.EQ, True, 0.7, {"features": {"self_ref": 0.12, "question_ratio": 0.5, "words": 22}, "valence": -0.6, "distress": 0.7}),
            make_result(Dimension.FRAGILITY, True, 0.7, {"pattern": "open", "pattern_scores": {"open": 0.8}}),
        ],
    },
}


class TestScenarioInsightQuality:
    """All scenarios should produce bloom-worthy insights."""

    @pytest.mark.parametrize("stype", SCENARIO_CONFIGS.keys())
    def test_scenario_bloom_worthy(self, stype):
        config = SCENARIO_CONFIGS[stype]
        text = config["text"]
        results = list(config["results"])  # copy to avoid mutation across runs
        # Fill missing dims
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(make_result(dim))

        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text=text)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]

        evaluator = InsightEvaluator()
        summary = evaluator.evaluate_batch(insights)

        print(f"\n--- {stype} ---")
        print(f"  Insights: {len(insights)} total, {len(good)} MEDIUM+")
        print(f"  Avg specificity: {summary.avg_specificity:.2f}")
        print(f"  Avg reframe quality: {summary.avg_reframe_quality:.2f}")
        print(f"  Avg overall: {summary.avg_overall:.2f}")
        print(f"  Bloom worthy: {summary.bloom_worthy}")
        for ins, ev in zip(insights, summary.per_insight):
            print(f"  [{ins.quality.name}] spec={ev.specificity:.2f} ref={ev.reframe_quality:.2f} | {ins.reframe[:70]}")
            if ev.flags:
                print(f"           FLAGS: {ev.flags}")

        # Core assertions
        assert len(good) >= 3, f"{stype}: only {len(good)} MEDIUM+ insights"
        assert summary.avg_reframe_quality >= 0.4, (
            f"{stype}: avg reframe quality too low ({summary.avg_reframe_quality:.2f})"
        )
        # No parrot-back flags
        all_flags = [f for ev in summary.per_insight for f in ev.flags]
        assert "parrot_back" not in all_flags, f"{stype}: has parrot-back insights"
