"""Base adapter interface for all detectors."""

from abc import ABC, abstractmethod

from sufficiency_scorer.config import MIN_CONFIDENCE
from sufficiency_scorer.models import DetectorResult, Dimension


class DetectorAdapter(ABC):
    """Adapter that wraps a detector and normalizes its output."""

    dimension: Dimension

    @abstractmethod
    async def detect(self, text: str, **kwargs) -> DetectorResult:
        """Run detection on text and return normalized result."""
        ...

    def _make_result(
        self, confidence: float, detail: dict | None = None
    ) -> DetectorResult:
        """Helper to build a result with automatic activation threshold."""
        activated = confidence >= MIN_CONFIDENCE
        return DetectorResult(
            dimension=self.dimension,
            activated=activated,
            confidence=min(max(confidence, 0.0), 1.0),
            detail=detail or {},
        )
