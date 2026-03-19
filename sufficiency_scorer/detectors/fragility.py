"""Fragility pattern detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

FRAGILITY_DETECTOR_PATH = Path.home() / "fragility-detector"


class FragilityAdapter(DetectorAdapter):
    """Adapts FragilityDetector.detect() → DetectorResult.

    Activation: pattern detected with confidence > 0.3.
    Confidence: detector's own confidence score.
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

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        conversation = [{"role": "user", "text": text}]
        snapshot = self._detector.detect(conversation=conversation, turn=1)
        confidence = getattr(snapshot, "confidence", 0.0)
        pattern = getattr(snapshot, "pattern", None)
        pattern_scores = getattr(snapshot, "pattern_scores", {})
        return self._make_result(
            confidence=confidence,
            detail={
                "pattern": str(pattern) if pattern else None,
                "pattern_scores": dict(pattern_scores),
            },
        )
