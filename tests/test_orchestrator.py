"""Tests for the orchestrator — mock detectors, verify parallel execution."""

import asyncio
import pytest

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.orchestrator import Orchestrator


class MockDetector(DetectorAdapter):
    """Mock detector that returns predetermined results."""

    def __init__(self, dimension: Dimension, activated: bool, confidence: float, delay: float = 0.0):
        self.dimension = dimension
        self._activated = activated
        self._confidence = confidence
        self._delay = delay

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._make_result(self._confidence if self._activated else 0.0)


class TestOrchestratorBasic:
    @pytest.mark.asyncio
    async def test_all_inactive(self):
        adapters = [MockDetector(dim, False, 0.0) for dim in Dimension]
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("test")
        assert report.score == 0.0
        assert report.activated_count == 0

    @pytest.mark.asyncio
    async def test_all_active(self):
        adapters = [MockDetector(dim, True, 0.8) for dim in Dimension]
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("test")
        assert report.score == 1.0
        assert report.ready is True

    @pytest.mark.asyncio
    async def test_partial_activation(self):
        active_dims = {Dimension.EMOTION, Dimension.EQ, Dimension.CONFLICT}
        adapters = [
            MockDetector(dim, dim in active_dims, 0.7)
            for dim in Dimension
        ]
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("test")
        assert 0.2 < report.score < 0.7
        assert report.activated_count == 3


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_faster_than_serial(self):
        """11 detectors with 0.1s delay each — parallel should be ~0.1s, not 1.1s."""
        adapters = [MockDetector(dim, True, 0.7, delay=0.1) for dim in Dimension]
        orch = Orchestrator(adapters=adapters)

        import time
        start = time.monotonic()
        report = await orch.evaluate("test")
        elapsed = time.monotonic() - start

        # Should be roughly 0.1s (parallel), not 1.1s (serial)
        assert elapsed < 0.5, f"Took {elapsed:.2f}s — detectors not running in parallel"
        assert report.score == 1.0


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_detector_doesnt_crash(self):
        """If one detector throws, others still work."""

        class FailingDetector(DetectorAdapter):
            dimension = Dimension.HUMOR

            async def detect(self, text: str, **kwargs) -> DetectorResult:
                raise RuntimeError("LLM unavailable")

        adapters = [
            MockDetector(dim, True, 0.7) if dim != Dimension.HUMOR
            else FailingDetector()
            for dim in Dimension
        ]
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("test")
        # Should still work, humor just not activated
        assert report.activated_count == 10  # all except humor
        humor_result = [r for r in report.detector_results if r.dimension == Dimension.HUMOR][0]
        assert humor_result.activated is False
        assert "error" in humor_result.detail
