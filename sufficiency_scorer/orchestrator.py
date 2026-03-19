"""Orchestrator — runs all detectors with parallelization and returns sufficiency score."""

import asyncio
import time

from sufficiency_scorer.config import PARALLEL_GROUPS
from sufficiency_scorer.detectors import ALL_ADAPTERS
from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension, SufficiencyReport
from sufficiency_scorer.scorer import SufficiencyScorer


class Orchestrator:
    """Runs all 11 detectors and computes ring sufficiency.

    Parallelization strategy:
      Group 0 (EQ): zero-cost behavioral features, runs first (<10ms)
      Group 1 (all LLM detectors): run in parallel (~1-2s total)

    Total latency target: < 2 seconds.
    """

    def __init__(self, adapters: list[DetectorAdapter] | None = None):
        if adapters is not None:
            self._adapters = {a.dimension: a for a in adapters}
        else:
            self._adapters = {cls.dimension: cls() for cls in ALL_ADAPTERS}
        self._scorer = SufficiencyScorer()

    async def evaluate(self, text: str, **kwargs) -> SufficiencyReport:
        """Run all detectors on text and return sufficiency report."""
        start = time.monotonic()
        results: list[DetectorResult] = []

        for group in PARALLEL_GROUPS:
            group_adapters = [
                self._adapters[dim] for dim in group if dim in self._adapters
            ]
            if not group_adapters:
                continue
            group_results = await asyncio.gather(
                *(adapter.detect(text, **kwargs) for adapter in group_adapters),
                return_exceptions=True,
            )
            for adapter, result in zip(group_adapters, group_results):
                if isinstance(result, Exception):
                    results.append(DetectorResult(
                        dimension=adapter.dimension,
                        activated=False,
                        confidence=0.0,
                        detail={"error": str(result)},
                    ))
                else:
                    results.append(result)

        # Fill in missing dimensions
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        elapsed = time.monotonic() - start
        report = self._scorer.score(results)
        return report
