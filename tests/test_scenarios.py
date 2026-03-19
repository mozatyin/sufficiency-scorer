"""Scenario validation — 9 types of user input with expected ring behavior.

Uses the 10_scenarios.json data from emotion-detector as ground truth for
emotion activation, and simulates the other detectors based on content analysis.
"""

import json
import pytest
from pathlib import Path

from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.scorer import SufficiencyScorer
from sufficiency_scorer.detectors.eq import EQAdapter


SCENARIOS_PATH = Path.home() / "emotion-detector" / "results" / "10_scenarios.json"


def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        pytest.skip("10_scenarios.json not found")
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


# Expected activation patterns per scenario type.
# These are based on the design doc's analysis of what each input type triggers.
EXPECTED_ACTIVATIONS: dict[str, dict[str, bool]] = {
    "工作压力": {
        "emotion": True, "eq": True, "conflict": True, "fragility": True,
        "humor": False, "mbti": False, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "自嘲幽默": {
        "emotion": True, "eq": True, "conflict": False, "fragility": True,
        "humor": True, "mbti": False, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "关系困境": {
        "emotion": True, "eq": True, "conflict": True, "fragility": True,
        "humor": False, "mbti": False, "love_language": True,
        "connection_response": True, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "自我探索": {
        "emotion": True, "eq": True, "conflict": False, "fragility": False,
        "humor": False, "mbti": True, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "焦虑失眠": {
        "emotion": True, "eq": True, "conflict": False, "fragility": True,
        "humor": False, "mbti": False, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "孤独隐形": {
        "emotion": True, "eq": True, "conflict": False, "fragility": True,
        "humor": False, "mbti": True, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "兴奋发现": {
        "emotion": True, "eq": True, "conflict": False, "fragility": False,
        "humor": False, "mbti": True, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "迷茫漂泊": {
        "emotion": True, "eq": True, "conflict": False, "fragility": True,
        "humor": False, "mbti": False, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
    "丧亲之痛": {
        "emotion": True, "eq": True, "conflict": False, "fragility": True,
        "humor": False, "mbti": False, "love_language": False,
        "connection_response": False, "character": False,
        "communication_dna": True, "soulgraph": True,
    },
}

# Expected score ranges per scenario type
EXPECTED_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "工作压力": (0.55, 0.90),    # emotion + conflict + fragility + eq + comm_dna + soulgraph = 6
    "自嘲幽默": (0.55, 0.90),    # emotion + humor + fragility + eq + comm_dna + soulgraph = 6
    "关系困境": (0.85, 1.00),    # 8 dimensions → should hit 100%
    "自我探索": (0.45, 0.80),    # emotion + eq + mbti + comm_dna + soulgraph = 5
    "焦虑失眠": (0.45, 0.80),    # emotion + eq + fragility + comm_dna + soulgraph = 5
    "孤独隐形": (0.55, 0.90),    # emotion + eq + fragility + mbti + comm_dna + soulgraph = 6
    "兴奋发现": (0.45, 0.80),    # emotion + eq + mbti + comm_dna + soulgraph = 5
    "迷茫漂泊": (0.45, 0.80),    # emotion + eq + fragility + comm_dna + soulgraph = 5
    "丧亲之痛": (0.45, 0.80),    # emotion + eq + fragility + comm_dna + soulgraph = 5
}


def simulate_detectors(scenario: dict, activations: dict[str, bool]) -> list[DetectorResult]:
    """Simulate detector results based on expected activation patterns.

    Uses real emotion data from 10_scenarios.json and simulates others.
    """
    results = []
    emotions = scenario.get("emotions", {})
    max_emotion = max(emotions.values()) if emotions else 0.0
    distress = scenario.get("distress", 0.0)

    dim_map = {
        "emotion": Dimension.EMOTION,
        "eq": Dimension.EQ,
        "conflict": Dimension.CONFLICT,
        "humor": Dimension.HUMOR,
        "mbti": Dimension.MBTI,
        "fragility": Dimension.FRAGILITY,
        "love_language": Dimension.LOVE_LANGUAGE,
        "connection_response": Dimension.CONNECTION_RESPONSE,
        "character": Dimension.CHARACTER,
        "communication_dna": Dimension.COMMUNICATION_DNA,
        "soulgraph": Dimension.SOULGRAPH,
    }

    for name, dim in dim_map.items():
        activated = activations.get(name, False)
        if name == "emotion":
            confidence = max_emotion if activated else 0.0
        elif name == "eq":
            confidence = min(1.0, distress + 0.3) if activated else 0.0
        elif name == "fragility":
            confidence = min(1.0, distress * 1.2) if activated else 0.0
        elif activated:
            confidence = 0.6  # default for simulated detectors
        else:
            confidence = 0.0

        results.append(DetectorResult(
            dimension=dim,
            activated=activated,
            confidence=confidence,
        ))

    return results


class TestScenarioScoring:
    """Validate that 9 scenario types produce reasonable ring progress."""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    @pytest.fixture
    def scenarios(self):
        return load_scenarios()

    def test_all_scenarios_in_expected_range(self, scorer, scenarios):
        for scenario in scenarios:
            stype = scenario["type"]
            if stype not in EXPECTED_ACTIVATIONS:
                continue
            activations = EXPECTED_ACTIVATIONS[stype]
            results = simulate_detectors(scenario, activations)
            report = scorer.score(results)
            lo, hi = EXPECTED_SCORE_RANGES[stype]
            assert lo <= report.score <= hi, (
                f"{stype}: score {report.score:.3f} not in [{lo}, {hi}]. "
                f"Activated: {report.activated_count}"
            )

    def test_work_pressure_activates_conflict(self, scorer, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "工作压力")
        activations = EXPECTED_ACTIVATIONS["工作压力"]
        results = simulate_detectors(scenario, activations)
        report = scorer.score(results)
        conflict_seg = [s for s in report.segments if s.dimension == Dimension.CONFLICT][0]
        assert conflict_seg.filled is True

    def test_self_deprecating_humor_activates_humor(self, scorer, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "自嘲幽默")
        activations = EXPECTED_ACTIVATIONS["自嘲幽默"]
        results = simulate_detectors(scenario, activations)
        report = scorer.score(results)
        humor_seg = [s for s in report.segments if s.dimension == Dimension.HUMOR][0]
        assert humor_seg.filled is True

    def test_relationship_crisis_near_full(self, scorer, scenarios):
        """关系困境 activates 8 dimensions → should be near/at 100%."""
        scenario = next(s for s in scenarios if s["type"] == "关系困境")
        activations = EXPECTED_ACTIVATIONS["关系困境"]
        results = simulate_detectors(scenario, activations)
        report = scorer.score(results)
        assert report.score >= 0.85
        assert report.activated_count >= 7

    def test_grief_activates_fragility(self, scorer, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "丧亲之痛")
        activations = EXPECTED_ACTIVATIONS["丧亲之痛"]
        results = simulate_detectors(scenario, activations)
        report = scorer.score(results)
        frag_seg = [s for s in report.segments if s.dimension == Dimension.FRAGILITY][0]
        assert frag_seg.filled is True


class TestGibberishRejection:
    """Garbage input should not move the ring."""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_hahaha_hi_zero_score(self, scorer):
        """'哈哈哈嗨' equivalent — nothing activates."""
        results = [DetectorResult(dimension=dim) for dim in Dimension]
        report = scorer.score(results)
        assert report.score == 0.0
        assert report.activated_count == 0
        assert report.prompt_hint == "tell_me_more"

    def test_only_eq_from_short_text(self, scorer):
        """Even if EQ picks up a faint signal, no emotion = capped."""
        results = [DetectorResult(dimension=dim) for dim in Dimension]
        results = [
            r if r.dimension != Dimension.EQ
            else DetectorResult(dimension=Dimension.EQ, activated=True, confidence=0.2)
            for r in results
        ]
        report = scorer.score(results)
        assert report.score <= 0.45
        assert report.prompt_hint == "how_do_you_feel"


class TestEQOnRealScenarios:
    """Run the real EQ adapter on scenario texts."""

    @pytest.fixture
    def scenarios(self):
        return load_scenarios()

    @pytest.mark.asyncio
    async def test_work_pressure_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "工作压力")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True
        assert result.detail["features"]["self_ref"] > 0.05

    @pytest.mark.asyncio
    async def test_excited_discovery_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "兴奋发现")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True
        # Should have positive emotion signal
        assert result.detail["features"]["pos_emotion_ratio"] > 0 or result.detail["features"]["exclamation_ratio"] > 0
