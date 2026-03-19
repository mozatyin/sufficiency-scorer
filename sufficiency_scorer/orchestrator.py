"""Orchestrator — manages session state, word gate, parallel detection, and scoring."""

import asyncio

from sufficiency_scorer.config import PARALLEL_GROUPS, MIN_WORDS
from sufficiency_scorer.detectors import ALL_ADAPTERS
from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension, SessionState, SufficiencyReport
from sufficiency_scorer.scorer import SufficiencyScorer


class Orchestrator:
    """Runs the full Touch ID flow: accumulate text -> check word gate -> run detectors -> score.

    Supports multi-press: each evaluate() call adds to the session.
    Call reset() to start a new session.
    """

    def __init__(self, adapters: list[DetectorAdapter] | None = None):
        if adapters is not None:
            self._adapters = {a.dimension: a for a in adapters}
        else:
            self._adapters = {cls.dimension: cls() for cls in ALL_ADAPTERS}
        self._scorer = SufficiencyScorer()
        self._session = SessionState(min_words=MIN_WORDS)

    def reset(self) -> None:
        """Clear session state for a new user interaction."""
        self._session = SessionState(min_words=MIN_WORDS)

    async def evaluate(self, text: str, **kwargs) -> SufficiencyReport:
        """Add text to session, check gate, run detectors if ready."""
        self._session.add_segment(text)

        if not self._session.meets_minimum:
            return SufficiencyReport(
                ready=False,
                insights=[],
                detector_results=[],
                ring_progress=0.0,
                prompt_hint="need_more_words",
            )

        full_text = self._session.full_text
        results = await self._run_detectors(full_text, **kwargs)
        return self._scorer.score(results)

    async def _run_detectors(self, text: str, **kwargs) -> list[DetectorResult]:
        """Run all detectors in parallel groups."""
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

        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        return results
