"""Conflict style detector adapter."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

CONFLICT_DETECTOR_PATH = Path.home() / "conflict-detector"


class ConflictAdapter(DetectorAdapter):
    """Adapts ConflictDetector.detect() → DetectorResult.

    Activation: any conflict style score > 0.3 (indicates interpersonal conflict context).
    Confidence: max style score.

    The underlying ConflictDetector uses synchronous anthropic SDK calls,
    so we wrap with asyncio.to_thread() to avoid blocking the event loop.
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

    def _run_sync(self, text: str):
        """Run the synchronous detector. Called via asyncio.to_thread()."""
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        snapshot = await asyncio.to_thread(self._run_sync, text)
        scores = snapshot.scores if hasattr(snapshot, "scores") else {}
        max_score = max(scores.values()) if scores else 0.0
        return self._make_result(
            confidence=max_score,
            detail={"styles": dict(scores)},
        )
