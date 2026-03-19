"""Emotion detector adapter — wraps emotion-detector project."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

EMOTION_DETECTOR_PATH = Path.home() / "emotion-detector"


class EmotionAdapter(DetectorAdapter):
    """Adapts EmotionDetector.detect() → DetectorResult.

    Activation: any emotion > 0.25 intensity.
    Confidence: max emotion intensity (strongest signal).

    The underlying EmotionDetector uses synchronous anthropic SDK calls,
    so we wrap with asyncio.to_thread() to avoid blocking the event loop.
    """

    dimension = Dimension.EMOTION

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(EMOTION_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(EMOTION_DETECTOR_PATH))
            from emotion_detector.detector import EmotionDetector
            self._detector = EmotionDetector()

    def _run_sync(self, text: str):
        """Run the synchronous detector. Called via asyncio.to_thread()."""
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        snapshot = await asyncio.to_thread(self._run_sync, text)
        return self._evaluate(snapshot)

    def _evaluate(self, snapshot) -> DetectorResult:
        emotions = snapshot.emotions if hasattr(snapshot, "emotions") else {}
        max_intensity = max(emotions.values()) if emotions else 0.0
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        return self._make_result(
            confidence=max_intensity,
            detail={"top_emotions": top_emotions},
        )
