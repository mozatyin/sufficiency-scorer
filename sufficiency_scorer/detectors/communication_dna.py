"""Communication DNA detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

COMM_DNA_PATH = Path.home() / "communication-dna"


class CommunicationDNAAdapter(DetectorAdapter):
    """Adapts CommunicationDNA.analyze() → DetectorResult.

    Activation: meaningful communication patterns detected.
    Confidence: based on number of features with signal.
    """

    dimension = Dimension.COMMUNICATION_DNA

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(COMM_DNA_PATH) not in sys.path:
                sys.path.insert(0, str(COMM_DNA_PATH))
            from communication_dna.detector import CommunicationDNADetector
            self._detector = CommunicationDNADetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        result = self._detector.analyze(
            text=text,
            speaker_id="user",
            speaker_label="User",
            context="soulmap_init",
        )
        features = getattr(result, "features", [])
        # Count features that have meaningful signal
        sig_features = [f for f in features if hasattr(f, "confidence") and f.confidence > 0.3]
        confidence = min(len(sig_features) / 15.0, 1.0) if sig_features else 0.0
        return self._make_result(
            confidence=confidence,
            detail={"feature_count": len(features), "significant": len(sig_features)},
        )
