"""Core scoring engine — determines readiness based on insight quality and count."""

from sufficiency_scorer.config import INSIGHT_THRESHOLD, MIN_INSIGHT_QUALITY_FOR_READY
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import (
    DetectorResult,
    SufficiencyReport,
)


class SufficiencyScorer:
    """Determines if we have enough high-quality insights to bloom.

    Logic:
      1. Extract insights from detector results (via InsightExtractor)
      2. Count insights at MEDIUM quality or above
      3. ready = (count >= INSIGHT_THRESHOLD)
      4. ring_progress = min(count / INSIGHT_THRESHOLD, 1.0)
    """

    def __init__(self):
        self._extractor = InsightExtractor()

    def score(self, results: list[DetectorResult]) -> SufficiencyReport:
        insights = self._extractor.extract(results)

        good_insights = [
            i for i in insights
            if i.quality.value >= MIN_INSIGHT_QUALITY_FOR_READY
        ]
        good_count = len(good_insights)

        ready = good_count >= INSIGHT_THRESHOLD
        ring_progress = min(good_count / INSIGHT_THRESHOLD, 1.0) if INSIGHT_THRESHOLD > 0 else 0.0
        if ready:
            ring_progress = 1.0

        if ready:
            prompt_hint = "ready"
        elif good_count == 0:
            prompt_hint = "tell_me_more"
        elif good_count >= INSIGHT_THRESHOLD - 1:
            prompt_hint = "almost_there"
        else:
            prompt_hint = "keep_going"

        return SufficiencyReport(
            ready=ready,
            insights=insights,
            detector_results=results,
            ring_progress=round(ring_progress, 4),
            prompt_hint=prompt_hint,
        )
