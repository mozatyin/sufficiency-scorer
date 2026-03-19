"""Emotion detector adapter — wraps emotion-detector project."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

EMOTION_DETECTOR_PATH = Path.home() / "emotion-detector"


class EmotionAdapter(DetectorAdapter):
    """Adapts EmotionDetector.detect() → DetectorResult.

    Activation: any emotion > 0.25 intensity.
    Confidence: max emotion intensity (strongest signal).
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

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        conversation = [{"role": "user", "text": text}]
        snapshot = await self._run(conversation)
        return self._evaluate(snapshot)

    async def _run(self, conversation: list[dict]):
        self._load()
        return self._detector.detect(conversation=conversation, turn=1)

    def _evaluate(self, snapshot) -> DetectorResult:
        emotions = snapshot.emotions if hasattr(snapshot, "emotions") else {}
        max_intensity = max(emotions.values()) if emotions else 0.0
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        return self._make_result(
            confidence=max_intensity,
            detail={"top_emotions": top_emotions},
        )
