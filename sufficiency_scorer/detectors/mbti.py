"""MBTI detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

MBTI_DETECTOR_PATH = Path.home() / "mbti-detector"


class MBTIAdapter(DetectorAdapter):
    """Adapts MBTIDetector.analyze() → DetectorResult.

    Activation: any dimension has confidence > 0.3.
    Confidence: average confidence across E/I + T/F dimensions.
    """

    dimension = Dimension.MBTI

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(MBTI_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(MBTI_DETECTOR_PATH))
            from mbti_detector.detector import MBTIDetector
            self._detector = MBTIDetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        result = self._detector.analyze(
            text=text,
            speaker_id="user",
            speaker_label="User",
            context="soulmap_init",
        )
        # Extract dimension confidences from PersonalityDNA
        traits = getattr(result, "traits", [])
        confidence_values = [t.confidence for t in traits if hasattr(t, "confidence") and t.confidence]
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return self._make_result(
            confidence=avg_confidence,
            detail={"trait_count": len(traits)},
        )
