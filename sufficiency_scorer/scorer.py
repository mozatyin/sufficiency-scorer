"""Core scoring engine — computes ring progress from detector results."""

from sufficiency_scorer.config import (
    WEIGHTS,
    EMOTION_GATE_CAP,
    ACTIVATION_TARGET,
    READY_THRESHOLD,
)
from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    RingSegment,
    SufficiencyReport,
)


class SufficiencyScorer:
    """Computes sufficiency score from detector results.

    Scoring formula:
      1. coverage = activated_count / ACTIVATION_TARGET (capped at 1.0)
      2. weighted_confidence = sum(weight_i * confidence_i) for activated dimensions
      3. score = coverage * 0.55 + weighted_confidence * 0.45
      4. If emotion not activated: score = min(score, EMOTION_GATE_CAP)
      5. If activated_count >= ACTIVATION_TARGET: score = max(score, 1.0)
    """

    def score(self, results: list[DetectorResult]) -> SufficiencyReport:
        """Compute sufficiency from a list of detector results."""
        activated = [r for r in results if r.activated]
        activated_count = len(activated)

        # Coverage: how many dimensions are lit
        coverage = min(activated_count / ACTIVATION_TARGET, 1.0)

        # Weighted confidence: quality of activated dimensions
        weighted_conf = sum(
            WEIGHTS.get(r.dimension, 0.0) * r.confidence
            for r in activated
        )
        # Normalize: max possible weighted_conf from activated dims
        max_possible = sum(WEIGHTS.get(r.dimension, 0.0) for r in activated)
        norm_conf = (weighted_conf / max_possible) if max_possible > 0 else 0.0

        # Combine
        raw_score = coverage * 0.55 + norm_conf * 0.45

        # Emotion gate
        emotion_activated = any(
            r.dimension == Dimension.EMOTION and r.activated for r in results
        )
        if not emotion_activated:
            raw_score = min(raw_score, EMOTION_GATE_CAP)

        # Hit target → force 100%
        if activated_count >= ACTIVATION_TARGET and emotion_activated:
            raw_score = 1.0

        final_score = round(min(max(raw_score, 0.0), 1.0), 4)
        ready = final_score >= READY_THRESHOLD

        # Build ring segments
        result_map = {r.dimension: r for r in results}
        segments = []
        for dim in Dimension:
            r = result_map.get(dim)
            segments.append(RingSegment(
                dimension=dim,
                filled=r.activated if r else False,
                intensity=r.confidence if r and r.activated else 0.0,
            ))

        # Prompt hint for UI
        if ready:
            prompt_hint = "ready"
        elif activated_count == 0:
            prompt_hint = "tell_me_more"
        elif not emotion_activated:
            prompt_hint = "how_do_you_feel"
        elif activated_count < 4:
            prompt_hint = "keep_going"
        else:
            prompt_hint = "almost_there"

        return SufficiencyReport(
            score=final_score,
            ready=ready,
            activated_count=activated_count,
            segments=segments,
            detector_results=results,
            prompt_hint=prompt_hint,
        )
