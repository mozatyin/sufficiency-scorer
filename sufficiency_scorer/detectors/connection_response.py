"""Connection response detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

CONNECTION_RESPONSE_PATH = Path.home() / "connection-response-detector"


class ConnectionResponseAdapter(DetectorAdapter):
    """Adapts ConnectionResponseDetector.classify() → DetectorResult.

    Activation: only when user text describes a bidirectional interaction.
    Note: per design doc, single-sentence init rarely activates this.
    """

    dimension = Dimension.CONNECTION_RESPONSE

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(CONNECTION_RESPONSE_PATH) not in sys.path:
                sys.path.insert(0, str(CONNECTION_RESPONSE_PATH))
            from connection_response.detector import ConnectionResponseDetector
            self._detector = ConnectionResponseDetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        # Connection response needs bidirectional conversation (bids + responses).
        # For init, we check if text describes an interaction pattern.
        # If it does, we extract a rough bid-response signal.
        self._load()
        conversation = [{"role": "user", "content": text, "turn": 1}]
        try:
            # Attempt to extract bids from user text
            bids = self._detector.extract_bids(conversation) if hasattr(self._detector, "extract_bids") else []
            if bids:
                responses = self._detector.classify(bids=bids, conversation=conversation)
                confidence = max(r.confidence for r in responses) if responses else 0.0
                patterns = [str(r.pattern) for r in responses]
            else:
                confidence = 0.0
                patterns = []
        except Exception:
            confidence = 0.0
            patterns = []
        return self._make_result(
            confidence=confidence,
            detail={"patterns": patterns},
        )
