"""Character traits detector adapter.

Note: per the design doc, Character (69 traits) is unreliable for short text —
it tends to over-diagnose from single sentences. We use a high activation
threshold to prevent false positives.
"""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

# Character detection reuses super-brain's architecture
SUPER_BRAIN_PATH = Path.home() / "super-brain"


class CharacterAdapter(DetectorAdapter):
    """Adapts SuperBrain personality analysis → DetectorResult.

    Activation: only if multiple traits detected with moderate confidence.
    Very conservative threshold to avoid the "tired → depression 0.99" problem.
    """

    dimension = Dimension.CHARACTER

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(SUPER_BRAIN_PATH) not in sys.path:
                sys.path.insert(0, str(SUPER_BRAIN_PATH))
            try:
                from super_brain.detector import PersonalityDetector
                self._detector = PersonalityDetector()
            except ImportError:
                self._detector = "unavailable"

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        if self._detector == "unavailable":
            return self._make_result(0.0, detail={"error": "super-brain not available"})

        try:
            result = self._detector.analyze(text=text, speaker_id="user")
            traits = getattr(result, "traits", [])
            # Conservative: only count traits with high confidence
            strong_traits = [t for t in traits if hasattr(t, "confidence") and t.confidence > 0.6]
            confidence = len(strong_traits) / 10.0 if strong_traits else 0.0
            confidence = min(confidence, 1.0)
            return self._make_result(
                confidence=confidence,
                detail={"strong_trait_count": len(strong_traits), "total_traits": len(traits)},
            )
        except Exception:
            return self._make_result(0.0, detail={"error": "detection failed"})
