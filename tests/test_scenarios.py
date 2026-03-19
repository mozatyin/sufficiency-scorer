"""Scenario validation — 9 types of user input with insight-based expectations.

Uses 10_scenarios.json from emotion-detector for real text inputs.
Tests that each scenario type produces the right number and type of insights.
"""

import json
import pytest
from pathlib import Path

from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality
from sufficiency_scorer.scorer import SufficiencyScorer
from sufficiency_scorer.detectors.eq import EQAdapter


SCENARIOS_PATH = Path.home() / "emotion-detector" / "results" / "10_scenarios.json"


def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        pytest.skip("10_scenarios.json not found")
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0, detail: dict | None = None) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


def simulate_scenario(activations: dict[str, dict]) -> list[DetectorResult]:
    """Build detector results from expected activation patterns."""
    results = []
    for dim in Dimension:
        spec = activations.get(dim.value)
        if spec and spec.get("activated"):
            results.append(make_result(dim, True, spec.get("confidence", 0.6), spec.get("detail", {})))
        else:
            results.append(make_result(dim))
    return results


SCENARIO_DETECTORS = {
    "工作压力": {
        "emotion": {"activated": True, "confidence": 0.7, "detail": {
            "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
        }},
        "eq": {"activated": True, "confidence": 0.6, "detail": {
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35, "distress": 0.43,
        }},
        "conflict": {"activated": True, "confidence": 0.6, "detail": {
            "styles": {"avoid": 0.7, "compromise": 0.4},
        }},
        "fragility": {"activated": True, "confidence": 0.5, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.6},
        }},
        "soulgraph": {"activated": True, "confidence": 0.5, "detail": {
            "items": 2, "avg_specificity": 0.5,
        }},
    },
    "自嘲幽默": {
        "emotion": {"activated": True, "confidence": 0.6, "detail": {
            "top_emotions": [("sadness", 0.5), ("amusement", 0.4)],
        }},
        "eq": {"activated": True, "confidence": 0.5, "detail": {
            "features": {"self_ref": 0.1, "question_ratio": 0.33, "words": 26},
            "valence": -0.15, "distress": 0.25,
        }},
        "humor": {"activated": True, "confidence": 0.7, "detail": {
            "humor_detected": True, "styles": {"self_deprecating": 0.8, "affiliative": 0.3},
        }},
        "conflict": {"activated": True, "confidence": 0.5, "detail": {
            "styles": {"avoid": 0.6, "compromise": 0.3},
        }},
        "fragility": {"activated": True, "confidence": 0.5, "detail": {
            "pattern": "masked", "pattern_scores": {"masked": 0.6},
        }},
        "soulgraph": {"activated": True, "confidence": 0.4, "detail": {
            "items": 1, "avg_specificity": 0.4,
        }},
    },
    "关系困境": {
        "emotion": {"activated": True, "confidence": 0.7, "detail": {
            "top_emotions": [("frustration", 0.6), ("sadness", 0.55), ("anger", 0.4)],
        }},
        "eq": {"activated": True, "confidence": 0.6, "detail": {
            "features": {"self_ref": 0.1, "question_ratio": 0.0, "words": 30},
            "valence": -0.4, "distress": 0.5,
        }},
        "conflict": {"activated": True, "confidence": 0.7, "detail": {
            "styles": {"avoid": 0.6, "collaborate": 0.5},
        }},
        "fragility": {"activated": True, "confidence": 0.6, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.7},
        }},
        "love_language": {"activated": True, "confidence": 0.5, "detail": {
            "has_relationship_context": True,
        }},
        "connection_response": {"activated": True, "confidence": 0.5, "detail": {
            "patterns": ["turning_away"],
        }},
        "soulgraph": {"activated": True, "confidence": 0.6, "detail": {
            "items": 3, "avg_specificity": 0.6,
        }},
    },
    "丧亲之痛": {
        "emotion": {"activated": True, "confidence": 0.8, "detail": {
            "top_emotions": [("grief", 0.8), ("sadness", 0.7), ("despair", 0.5)],
        }},
        "eq": {"activated": True, "confidence": 0.7, "detail": {
            "features": {"self_ref": 0.12, "question_ratio": 0.0, "words": 35},
            "valence": -0.6, "distress": 0.7,
        }},
        "fragility": {"activated": True, "confidence": 0.7, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.8},
        }},
        "soulgraph": {"activated": True, "confidence": 0.5, "detail": {
            "items": 2, "avg_specificity": 0.5,
        }},
    },
}


class TestScenarioInsights:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_work_pressure_ready(self, scorer):
        results = simulate_scenario(SCENARIO_DETECTORS["工作压力"])
        report = scorer.score(results)
        assert report.ready is True
        assert len(report.insights) >= 3

    def test_self_deprecating_humor_ready(self, scorer):
        results = simulate_scenario(SCENARIO_DETECTORS["自嘲幽默"])
        report = scorer.score(results)
        assert report.ready is True
        humor_insights = [i for i in report.insights if Dimension.HUMOR in i.source_dimensions]
        assert len(humor_insights) >= 1

    def test_relationship_crisis_ready_with_many(self, scorer):
        results = simulate_scenario(SCENARIO_DETECTORS["关系困境"])
        report = scorer.score(results)
        assert report.ready is True
        assert len(report.insights) >= 4

    def test_grief_ready(self, scorer):
        results = simulate_scenario(SCENARIO_DETECTORS["丧亲之痛"])
        report = scorer.score(results)
        assert report.ready is True
        assert any("grief" in i.signal or "loss" in i.reframe.lower() or "depth" in i.reframe.lower()
                    for i in report.insights)


class TestGibberishRejection:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_nothing_activated_not_ready(self, scorer):
        results = [make_result(dim) for dim in Dimension]
        report = scorer.score(results)
        assert report.ready is False
        assert len(report.insights) == 0
        assert report.prompt_hint == "tell_me_more"


class TestEQOnRealScenarios:
    @pytest.fixture
    def scenarios(self):
        return load_scenarios()

    @pytest.mark.asyncio
    async def test_work_pressure_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "工作压力")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True

    @pytest.mark.asyncio
    async def test_excited_discovery_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "兴奋发现")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True
