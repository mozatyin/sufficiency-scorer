"""V8x Star Generation Logic — with minimum guarantee.

Decides when to generate new stars based on detector results,
enforces minimum star counts per turn, and falls back to
user topics when detectors don't fire.
"""

from __future__ import annotations

from sufficiency_scorer.v8x_labels import get_star_label, LABEL_MAP

# ── Minimum guarantee table ─────────────────────────────────────────
# turn_number -> minimum star count required
MINIMUM_STARS: dict[int, int] = {
    1: 1,
    2: 2,
    4: 3,
    6: 4,
    10: 5,
}

# Topic -> dimension fallback mapping
TOPIC_DIMENSION_MAP: dict[str, str] = {
    "work": "behavioral",
    "career": "behavioral",
    "stress": "emotion",
    "family": "love_language",
    "relationships": "attachment",
    "friends": "connection_response",
    "anger": "emotion",
    "love": "love_language",
    "conflict": "conflict",
    "music": "values",
    "art": "values",
    "humor": "humor",
    "funny": "humor",
}

# Ordered fallback dimensions when we need to pick one
FALLBACK_DIMENSIONS: list[str] = [
    "emotion", "conflict", "humor", "mbti", "love_language",
    "eq", "fragility", "attachment", "values", "behavioral",
    "connection_response",
]

# Default detail payloads for fallback star generation
_FALLBACK_DETAILS: dict[str, dict] = {
    "emotion": {"frustration": 0.6},
    "conflict": {"style": "avoid"},
    "humor": {"style": "affiliative"},
    "mbti": {"I": 0.7},
    "love_language": {"primary": "words"},
    "eq": {"empathy": 0.8},
    "fragility": {"pattern": "open"},
    "attachment": {"style": "secure"},
    "values": {"primary": "benevolence"},
    "behavioral": {"question_ratio": 0.3},
    "connection_response": {"style": "connected"},
}


class StarGenerator:
    """Generates stars from detector results with minimum guarantees."""

    def should_generate_star(
        self,
        detector_result: dict,
        existing_stars: list[str],
        turn_number: int,
    ) -> dict:
        """Decide whether a single detector result should produce a new star.

        Returns:
            {"generate": bool, "dimension": str, "trigger": str, "star_type": str}
        """
        dimension = detector_result.get("dimension", "")
        confidence = detector_result.get("confidence", 0.0)

        # Rule 1: confidence must exceed 0.5
        if confidence <= 0.5:
            return {"generate": False, "dimension": dimension, "trigger": "low_confidence", "star_type": ""}

        # Rule 2: dimension must not already exist
        if dimension in existing_stars:
            return {"generate": False, "dimension": dimension, "trigger": "already_exists", "star_type": ""}

        # Generate — resolve label to get star_type
        detail = detector_result.get("detail", {})
        label = get_star_label(dimension, detail)
        star_type = label.star_type if label else "bright"

        return {
            "generate": True,
            "dimension": dimension,
            "trigger": "new_dimension",
            "star_type": star_type,
        }

    def enforce_minimum_guarantee(
        self,
        existing_stars: list[str],
        turn_number: int,
        detector_results: list[dict],
        user_topics: list[str],
    ) -> list[dict]:
        """Force-generate stars if minimum count not met.

        Returns list of star dicts to add.
        """
        # Find the applicable minimum for this turn
        required = 0
        for t in sorted(MINIMUM_STARS.keys()):
            if turn_number >= t:
                required = MINIMUM_STARS[t]

        deficit = required - len(existing_stars)
        if deficit <= 0:
            return []

        forced: list[dict] = []
        used_dims = set(existing_stars)

        # Strategy 1: try to pick from detector results (even low-confidence ones)
        candidates = sorted(detector_results, key=lambda r: r.get("confidence", 0), reverse=True)
        for dr in candidates:
            if deficit <= 0:
                break
            dim = dr.get("dimension", "")
            if dim and dim not in used_dims:
                star = self._make_star(dim, dr.get("detail", _FALLBACK_DETAILS.get(dim, {})))
                if star:
                    forced.append(star)
                    used_dims.add(dim)
                    deficit -= 1

        # Strategy 2: fallback to user topics
        if deficit > 0 and user_topics:
            for topic in user_topics:
                if deficit <= 0:
                    break
                topic_lower = topic.lower()
                dim = None
                for keyword, mapped_dim in TOPIC_DIMENSION_MAP.items():
                    if keyword in topic_lower:
                        dim = mapped_dim
                        break
                if dim is None:
                    # Map unknown topics to first available dimension
                    dim = self._next_available_dimension(used_dims)
                if dim and dim not in used_dims:
                    star = self._make_star(dim, _FALLBACK_DETAILS.get(dim, {}))
                    if star:
                        forced.append(star)
                        used_dims.add(dim)
                        deficit -= 1

        # Strategy 3: pick next untagged dimension from fallback list
        if deficit > 0:
            for dim in FALLBACK_DIMENSIONS:
                if deficit <= 0:
                    break
                if dim not in used_dims:
                    star = self._make_star(dim, _FALLBACK_DETAILS.get(dim, {}))
                    if star:
                        forced.append(star)
                        used_dims.add(dim)
                        deficit -= 1

        return forced

    def generate_stars_for_turn(
        self,
        detector_results: list[dict],
        existing_stars: list[str],
        turn_number: int,
        user_topics: list[str] | None = None,
    ) -> list[dict]:
        """Main entry point. Combines should_generate_star + enforce_minimum_guarantee.

        Returns list of star dicts with:
            dimension, label, sublabel, star_type, star_color, confidence, is_question
        """
        user_topics = user_topics or []
        generated: list[dict] = []

        # Phase 1: evaluate each detector result, pick at most 1 (highest confidence)
        candidates = []
        for dr in detector_results:
            verdict = self.should_generate_star(dr, existing_stars, turn_number)
            if verdict["generate"]:
                candidates.append(dr)

        if candidates:
            # Pick highest confidence
            best = max(candidates, key=lambda r: r.get("confidence", 0))
            star = self._make_star(
                best["dimension"],
                best.get("detail", {}),
                confidence=best.get("confidence", 0.0),
            )
            if star:
                generated.append(star)

        # Phase 2: enforce minimum guarantee
        current_dims = list(existing_stars) + [s["dimension"] for s in generated]
        forced = self.enforce_minimum_guarantee(
            existing_stars=current_dims,
            turn_number=turn_number,
            detector_results=detector_results,
            user_topics=user_topics,
        )
        generated.extend(forced)

        return generated

    # ── Private helpers ──────────────────────────────────────────────

    def _make_star(self, dimension: str, detail: dict, confidence: float = 0.5) -> dict | None:
        """Build a star dict using v8x_labels."""
        label = get_star_label(dimension, detail)
        if label is None:
            return None
        return {
            "dimension": dimension,
            "label": label.label,
            "sublabel": label.sublabel,
            "star_type": label.star_type,
            "star_color": label.star_color,
            "confidence": confidence,
            "is_question": label.is_question,
        }

    @staticmethod
    def _next_available_dimension(used: set[str]) -> str | None:
        """Return the first fallback dimension not yet used."""
        for dim in FALLBACK_DIMENSIONS:
            if dim not in used:
                return dim
        return None
