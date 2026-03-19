"""Conflict style detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

CONFLICT_DETECTOR_PATH = Path.home() / "conflict-detector"


class ConflictAdapter(DetectorAdapter):
    """Adapts ConflictDetector.detect() → DetectorResult.

    Activation: any conflict style score > 0.3 (indicates interpersonal conflict context).
    Confidence: max style score.
    """

    dimension = Dimension.CONFLICT

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(CONFLICT_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(CONFLICT_DETECTOR_PATH))
            from conflict_detector.detector import ConflictDetector
            self._detector = ConflictDetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        conversation = [{"role": "user", "text": text}]
        snapshot = self._detector.detect(conversation=conversation, turn=1)
        scores = snapshot.scores if hasattr(snapshot, "scores") else {}
        max_score = max(scores.values()) if scores else 0.0
        return self._make_result(
            confidence=max_score,
            detail={"styles": dict(scores)},
        )
