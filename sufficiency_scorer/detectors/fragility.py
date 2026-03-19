"""Fragility pattern detector adapter."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

FRAGILITY_DETECTOR_PATH = Path.home() / "fragility-detector"


class FragilityAdapter(DetectorAdapter):
    """Adapts FragilityDetector.detect() → DetectorResult.

    Activation: pattern detected with confidence > 0.3.
    Confidence: detector's own confidence score.

    The underlying FragilityDetector uses synchronous anthropic SDK calls,
    so we wrap with asyncio.to_thread() to avoid blocking the event loop.
    FragilityPattern is a str Enum — we call .value for serialization.
    """

    dimension = Dimension.FRAGILITY

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(FRAGILITY_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(FRAGILITY_DETECTOR_PATH))
            from fragility_detector.detector import FragilityDetector
            self._detector = FragilityDetector()

    def _run_sync(self, text: str):
        """Run the synchronous detector. Called via asyncio.to_thread()."""
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        snapshot = await asyncio.to_thread(self._run_sync, text)
        confidence = getattr(snapshot, "confidence", 0.0)
        pattern = getattr(snapshot, "pattern", None)
        pattern_scores = getattr(snapshot, "pattern_scores", {})
        # FragilityPattern is a str Enum — extract .value for clean serialization
        pattern_name = pattern.value if hasattr(pattern, "value") else str(pattern) if pattern else None
        return self._make_result(
            confidence=confidence,
            detail={
                "pattern": pattern_name,
                "pattern_scores": dict(pattern_scores),
            },
        )
