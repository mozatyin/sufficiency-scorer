"""Star Engine — Gap 4 + Gap 6.

Decides when to create stars, what to label them, and emits fog events.

Gap 4: Star generation logic
  - New dimension first activation (confidence > threshold) → new star
  - Existing dimension confidence change → brightness change (no new star)
  - Minimum guarantee: turn 2 ≥ 2 stars, turn 4 ≥ 3 stars
  - Fallback: extract most obvious unlabeled dimension from user text
  - Max 1 new star per turn

Gap 6: Fog disturbance signal
  - Detector identifies new dimension → emit fog_event immediately
  - {position: [x,y], intensity: 0-1, eta_ms: 3000}
  - 3 seconds later → star_created event
"""

import random
import time
from pydantic import BaseModel, Field

from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.star_labels import (
    get_positive_label,
    get_signal_key,
    get_dark_labels,
    BANNED_TERMS,
)


# === Configuration ===
ACTIVATION_THRESHOLD = 0.20  # minimum confidence to count as activated
BRIGHTNESS_CHANGE_THRESHOLD = 0.15  # min delta to trigger brightness update
FOG_ETA_MS = 3000  # fog → star delay
MAX_NEW_STARS_PER_TURN = 1
MIN_STARS_BY_TURN = {2: 2, 3: 2, 4: 3}  # turn → minimum star count


# === Data Models ===

class Star(BaseModel):
    """A star on the soul map."""
    id: str
    dimension: Dimension
    signal_key: str
    label: str
    is_dark: bool = Field(default=False, description="30% dark star with ?")
    brightness: float = Field(default=0.7, ge=0.0, le=1.0)
    created_at_turn: int = 0


class FogEvent(BaseModel):
    """Fog disturbance — precedes star creation by 3 seconds."""
    position: list[float] = Field(description="[x, y] on star map, 0.0-1.0")
    intensity: float = Field(ge=0.0, le=1.0)
    eta_ms: int = FOG_ETA_MS
    dimension: Dimension


class StarCreatedEvent(BaseModel):
    """Star materialized after fog clears."""
    star: Star
    fog_position: list[float]


class BrightnessChangeEvent(BaseModel):
    """Existing star brightness changed (no new star)."""
    star_id: str
    old_brightness: float
    new_brightness: float


class StarEngineOutput(BaseModel):
    """Full output of one turn of the star engine."""
    fog_events: list[FogEvent] = Field(default_factory=list)
    new_stars: list[StarCreatedEvent] = Field(default_factory=list)
    brightness_changes: list[BrightnessChangeEvent] = Field(default_factory=list)
    total_stars: int = 0


# === Star Engine ===

class StarEngine:
    """Manages the star map across conversation turns.

    Usage:
        engine = StarEngine()
        output = engine.process_turn(detector_results, turn_count=1)
        # output.fog_events → send to frontend immediately
        # after 3s → output.new_stars → render stars
    """

    def __init__(self, label_generator=None):
        self.stars: list[Star] = []
        self._activated_dims: dict[Dimension, float] = {}  # dim → last confidence
        self._label_generator = label_generator
        self._user_text = ""

    def process_turn(
        self,
        results: list[DetectorResult],
        turn_count: int,
        user_text: str = "",
    ) -> StarEngineOutput:
        """Process one conversation turn. Returns fog/star/brightness events."""
        output = StarEngineOutput()
        self._user_text = user_text

        activated = [r for r in results if r.activated and r.confidence >= ACTIVATION_THRESHOLD]
        new_dim_candidates: list[DetectorResult] = []
        brightness_updates: list[DetectorResult] = []

        for r in activated:
            if r.dimension not in self._activated_dims:
                # New dimension — candidate for new star
                new_dim_candidates.append(r)
            else:
                # Existing dimension — check brightness change
                old_conf = self._activated_dims[r.dimension]
                delta = abs(r.confidence - old_conf)
                if delta >= BRIGHTNESS_CHANGE_THRESHOLD:
                    brightness_updates.append(r)

        # Sort candidates: prefer dimensions we don't have yet, then by confidence
        existing_dims = {s.dimension for s in self.stars}
        new_dim_candidates.sort(key=lambda r: (
            r.dimension not in existing_dims,  # True sorts after False, so negate
            r.confidence,
        ), reverse=True)

        # Create at most 1 new star
        new_star_created = False
        if new_dim_candidates:
            top = new_dim_candidates[0]
            star = self._create_star(top, turn_count)
            if star:
                pos = self._random_position()
                output.fog_events.append(FogEvent(
                    position=pos,
                    intensity=min(top.confidence * 1.2, 1.0),
                    dimension=top.dimension,
                ))
                output.new_stars.append(StarCreatedEvent(star=star, fog_position=pos))
                new_star_created = True

        # Process brightness changes
        for r in brightness_updates:
            matching = [s for s in self.stars if s.dimension == r.dimension]
            if matching:
                star = matching[0]
                old_b = star.brightness
                star.brightness = round(min(max(r.confidence, 0.1), 1.0), 2)
                output.brightness_changes.append(BrightnessChangeEvent(
                    star_id=star.id,
                    old_brightness=old_b,
                    new_brightness=star.brightness,
                ))

        # Update tracking
        for r in activated:
            self._activated_dims[r.dimension] = r.confidence

        # Minimum guarantee fallback — keep adding until we meet the minimum
        min_required = MIN_STARS_BY_TURN.get(turn_count, 0)
        while len(self.stars) < min_required:
            fallback = self._fallback_star(results, turn_count, user_text)
            if fallback:
                pos = self._random_position()
                output.fog_events.append(FogEvent(
                    position=pos,
                    intensity=0.5,
                    dimension=fallback.dimension,
                ))
                output.new_stars.append(StarCreatedEvent(star=fallback, fog_position=pos))
            else:
                break  # no more fallback candidates

        # Check dark star opportunity (at most 1 per session, after turn 3)
        if turn_count >= 3 and not any(s.is_dark for s in self.stars):
            dark = self._try_dark_star(results, turn_count)
            if dark:
                pos = self._random_position()
                output.fog_events.append(FogEvent(
                    position=pos,
                    intensity=0.3,
                    dimension=Dimension.EMOTION,
                ))
                output.new_stars.append(StarCreatedEvent(star=dark, fog_position=pos))

        output.total_stars = len(self.stars)
        return output

    def _create_star(self, result: DetectorResult, turn: int) -> Star | None:
        """Create a bright (70%) star from a detector result."""
        signal_key = get_signal_key(result.dimension, result.detail)
        if not signal_key:
            return None

        # Try LLM-generated label if generator available, else fall back to template
        label = None
        if self._label_generator and self._user_text:
            label = self._label_generator.generate_label(
                result.dimension.value, signal_key, self._user_text
            )
        if not label:
            label = get_positive_label(result.dimension, signal_key)
        if not label:
            return None

        # Safety check
        if any(banned in label for banned in BANNED_TERMS):
            return None
        # Dedup: don't create a star with a label we already have
        existing_labels = {s.label for s in self.stars}
        if label in existing_labels:
            # Try template fallback instead
            template_label = get_positive_label(result.dimension, signal_key)
            if template_label and template_label not in existing_labels:
                label = template_label
            else:
                return None
        star = Star(
            id=f"star_{result.dimension.value}_{len(self.stars)}",
            dimension=result.dimension,
            signal_key=signal_key,
            label=label,
            brightness=round(min(result.confidence, 1.0), 2),
            created_at_turn=turn,
        )
        self.stars.append(star)
        self._activated_dims[result.dimension] = result.confidence
        return star

    def _fallback_star(
        self, results: list[DetectorResult], turn: int, user_text: str
    ) -> Star | None:
        """Fallback: create star from most obvious unlabeled dimension."""
        existing_dims = {s.dimension for s in self.stars}
        # Find activated but un-starred dimensions (very low threshold for fallback)
        candidates = [
            r for r in results
            if r.activated and r.dimension not in existing_dims and r.confidence >= 0.05
        ]
        candidates.sort(key=lambda r: r.confidence, reverse=True)
        for c in candidates:
            star = self._create_star(c, turn)
            if star:
                return star

        # Ultra-fallback: EQ is almost always activated
        if Dimension.EQ not in existing_dims:
            for r in results:
                if r.dimension == Dimension.EQ and r.activated:
                    return self._create_star(r, turn)

        # Last resort: create a star from topic/text heuristic
        # Pick an un-starred dimension and give it a generic but relevant label
        unstarred = [d for d in Dimension if d not in existing_dims
                     and d not in (Dimension.CHARACTER, Dimension.COMMUNICATION_DNA, Dimension.CONNECTION_RESPONSE)]
        if unstarred and user_text:
            from sufficiency_scorer.star_labels import POSITIVE_LABELS
            for dim in unstarred:
                labels = POSITIVE_LABELS.get(dim, {})
                if labels:
                    # Pick the first label that fits
                    first_key = next(iter(labels))
                    star = Star(
                        id=f"star_{dim.value}_{len(self.stars)}",
                        dimension=dim,
                        signal_key=first_key,
                        label=labels[first_key],
                        brightness=0.4,
                        created_at_turn=turn,
                    )
                    self.stars.append(star)
                    self._activated_dims[dim] = 0.3
                    return star
        return None

    def _try_dark_star(self, results: list[DetectorResult], turn: int) -> Star | None:
        """Try to create a 30% dark star from cross-dimensional patterns."""
        result_map = {}
        for r in results:
            if r.activated:
                result_map[r.dimension] = {"detail": r.detail, "confidence": r.confidence}

        dark_labels = get_dark_labels(result_map)
        if dark_labels:
            label = dark_labels[0]
            star = Star(
                id=f"star_dark_{len(self.stars)}",
                dimension=Dimension.EMOTION,  # dark stars don't have single dimension
                signal_key="dark",
                label=label,
                is_dark=True,
                brightness=0.3,
                created_at_turn=turn,
            )
            self.stars.append(star)
            return star
        return None

    @staticmethod
    def _random_position() -> list[float]:
        """Random position on the star map (0.0-1.0 range)."""
        return [round(random.uniform(0.15, 0.85), 3), round(random.uniform(0.15, 0.85), 3)]

    def reset(self) -> None:
        """Clear all state for a new session."""
        self.stars.clear()
        self._activated_dims.clear()
