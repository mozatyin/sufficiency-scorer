"""Love language detector adapter."""

import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

LOVE_LANGUAGE_PATH = Path.home() / "love-language-detector"


class LoveLanguageAdapter(DetectorAdapter):
    """Adapts LoveLanguageDetector.detect() → DetectorResult.

    Activation: has_relationship_context is True.
    Confidence: max Chapman score.
    Note: per the design doc, this rarely activates from short init text
    (needs relationship context). That's correct behavior.
    """

    dimension = Dimension.LOVE_LANGUAGE

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(LOVE_LANGUAGE_PATH) not in sys.path:
                sys.path.insert(0, str(LOVE_LANGUAGE_PATH))
            from love_language.detector import LoveLanguageDetector
            self._detector = LoveLanguageDetector()

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self._load()
        messages = [{"role": "user", "content": text}]
        result = self._detector.detect(messages=messages)
        has_context = getattr(result, "has_relationship_context", False)
        chapman = getattr(result, "chapman", None)
        if chapman and has_context:
            scores = {k: v for k, v in chapman.__dict__.items() if isinstance(v, (int, float))}
            max_score = max(scores.values()) if scores else 0.0
        else:
            scores = {}
            max_score = 0.0
        confidence = max_score if has_context else 0.0
        return self._make_result(
            confidence=confidence,
            detail={"has_relationship_context": has_context, "chapman": scores},
        )
