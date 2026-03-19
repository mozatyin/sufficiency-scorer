"""Tests for the orchestrator — session accumulation, word gate, parallel execution."""

import asyncio
import pytest

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.orchestrator import Orchestrator


class MockDetector(DetectorAdapter):
    def __init__(self, dimension: Dimension, activated: bool, confidence: float, detail: dict | None = None, delay: float = 0.0):
        self.dimension = dimension
        self._activated = activated
        self._confidence = confidence
        self._detail = detail or {}
        self._delay = delay
        self.call_count = 0

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return DetectorResult(
            dimension=self.dimension,
            activated=self._activated,
            confidence=self._confidence if self._activated else 0.0,
            detail=self._detail,
        )


def make_rich_mocks() -> list[MockDetector]:
    """Mocks that simulate a rich emotional input."""
    active = {
        Dimension.EMOTION: {"confidence": 0.7, "detail": {"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}},
        Dimension.EQ: {"confidence": 0.6, "detail": {"features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27}, "valence": -0.35, "distress": 0.43}},
        Dimension.CONFLICT: {"confidence": 0.6, "detail": {"styles": {"avoid": 0.7, "compromise": 0.4}}},
        Dimension.FRAGILITY: {"confidence": 0.5, "detail": {"pattern": "open", "pattern_scores": {"open": 0.6}}},
        Dimension.SOULGRAPH: {"confidence": 0.5, "detail": {"items": 2, "avg_specificity": 0.5}},
    }
    mocks = []
    for dim in Dimension:
        if dim in active:
            spec = active[dim]
            mocks.append(MockDetector(dim, True, spec["confidence"], spec["detail"]))
        else:
            mocks.append(MockDetector(dim, False, 0.0))
    return mocks


class TestWordGate:
    @pytest.mark.asyncio
    async def test_under_40_words_not_ready(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("Short text here.")
        assert report.ready is False
        assert report.prompt_hint == "need_more_words"

    @pytest.mark.asyncio
    async def test_under_40_words_detectors_not_called(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        await orch.evaluate("Short text.")
        for adapter in adapters:
            assert adapter.call_count == 0

    @pytest.mark.asyncio
    async def test_over_40_words_runs_detectors(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        long_text = (
            "I'm really stressed about work lately. My boss keeps pushing me "
            "to do overtime and I don't know how to say no. I feel like I'm "
            "losing myself in this job and I can't figure out what to do anymore. "
            "Every day feels the same and I'm exhausted."
        )
        report = await orch.evaluate(long_text)
        assert any(a.call_count > 0 for a in adapters)


class TestSessionAccumulation:
    @pytest.mark.asyncio
    async def test_cumulative_text_passes_gate(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        r1 = await orch.evaluate("My boss makes me work overtime.")
        assert r1.ready is False
        r2 = await orch.evaluate(
            "I feel trapped and I don't know what to do. "
            "Every day is the same and I'm losing motivation. "
            "I used to love what I do but now it feels pointless. "
            "I really need some help figuring this out."
        )
        assert any(a.call_count > 0 for a in adapters)

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        long_text = " ".join(["word"] * 50)
        await orch.evaluate(long_text)
        orch.reset()
        r = await orch.evaluate("Short.")
        assert r.ready is False
        assert r.prompt_hint == "need_more_words"


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_faster_than_serial(self):
        adapters = [MockDetector(dim, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)] if dim == Dimension.EMOTION else [],
            "styles": {"avoid": 0.7} if dim == Dimension.CONFLICT else {},
            "features": {"self_ref": 0.1, "question_ratio": 0.2, "words": 50} if dim == Dimension.EQ else {},
            "valence": -0.3 if dim == Dimension.EQ else 0.0,
            "distress": 0.4 if dim == Dimension.EQ else 0.0,
            "pattern": "open" if dim == Dimension.FRAGILITY else None,
            "humor_detected": True if dim == Dimension.HUMOR else False,
            "items": 2 if dim == Dimension.SOULGRAPH else 0,
        }, delay=0.1) for dim in Dimension]
        orch = Orchestrator(adapters=adapters)

        import time
        long_text = " ".join(["word"] * 50)
        start = time.monotonic()
        await orch.evaluate(long_text)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — detectors not running in parallel"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_detector_doesnt_crash(self):
        class FailingDetector(DetectorAdapter):
            dimension = Dimension.HUMOR
            async def detect(self, text: str, **kwargs) -> DetectorResult:
                raise RuntimeError("LLM unavailable")

        adapters = [
            MockDetector(dim, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55)] if dim == Dimension.EMOTION else [],
            }) if dim != Dimension.HUMOR
            else FailingDetector()
            for dim in Dimension
        ]
        orch = Orchestrator(adapters=adapters)
        long_text = " ".join(["word"] * 50)
        report = await orch.evaluate(long_text)
        humor_result = [r for r in report.detector_results if r.dimension == Dimension.HUMOR]
        assert len(humor_result) == 1
        assert humor_result[0].activated is False
