"""Humor style detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

HUMOR_DETECTOR_PATH = Path.home() / "humor-detector"


class HumorAdapter(DetectorAdapter):
    """Adapts HumorDetector.detect() → DetectorResult.

    Activation: humor_detected is True.
    Confidence: amusement_intensity or max style score.
    """

    dimension = Dimension.HUMOR

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(HUMOR_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(HUMOR_DETECTOR_PATH))
            from humor_detector.detector import HumorDetector
            self._detector = HumorDetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        snapshot = self._detector.detect(text=text, turn=0)
        humor_detected = getattr(snapshot, "humor_detected", False)
        amusement = getattr(snapshot, "amusement_intensity", 0.0)
        styles = getattr(snapshot, "styles", {})
        max_style = max(styles.values()) if styles else 0.0
        confidence = max(amusement, max_style) if humor_detected else 0.0
        return self._make_result(
            confidence=confidence,
            detail={"humor_detected": humor_detected, "styles": dict(styles)},
        )
