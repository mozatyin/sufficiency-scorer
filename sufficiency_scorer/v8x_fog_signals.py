"""V8x Fog Signal Emitter — FR-08.

Generates fog_disturbance and star_created signals for the SoulMap
constellation frontend. When a new star is about to appear, the fog
stirs first (3s lead time), then the star materializes.
"""

import hashlib
import math
import time


class FogSignalEmitter:
    """Emits fog disturbance and star creation signals for the frontend."""

    FOG_ETA_MS = 3000
    MIN_STAR_DISTANCE = 0.15

    def create_fog_disturbance(
        self,
        dimension: str,
        confidence: float,
        existing_stars: list,
    ) -> dict:
        """Create a fog_disturbance signal for an emerging star."""
        position = self.calculate_position(dimension, existing_stars)
        return {
            "event": "fog_disturbance",
            "dimension": dimension,
            "position": position,
            "intensity": confidence,
            "will_become_star": True,
            "eta_ms": self.FOG_ETA_MS,
            "timestamp_ms": _now_ms(),
        }

    def create_star_created(
        self,
        dimension: str,
        label: str,
        star_color: str,
        position: dict,
    ) -> dict:
        """Create a star_created signal after fog disturbance completes."""
        return {
            "event": "star_created",
            "dimension": dimension,
            "label": label,
            "star_color": star_color,
            "position": position,
            "timestamp_ms": _now_ms(),
        }

    def calculate_position(
        self,
        dimension: str,
        existing_stars: list,
    ) -> dict:
        """Calculate normalized (0-1) x,y position for a new star.

        - First star goes to center (0.5, 0.5).
        - Subsequent stars use a deterministic hash-based angle, then
          nudge outward until minimum distance from all existing stars
          is satisfied.
        """
        if not existing_stars:
            return {"x": 0.5, "y": 0.5}

        # Deterministic angle from dimension name
        angle = _hash_to_angle(dimension)
        radius = 0.2  # starting radius from center

        existing_positions = [
            (s["position"]["x"], s["position"]["y"]) for s in existing_stars
        ]

        # Try increasing radius until we find a non-overlapping spot
        for _ in range(20):
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            # Clamp to [0.05, 0.95] so stars stay visible
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            if _min_distance(x, y, existing_positions) >= self.MIN_STAR_DISTANCE:
                return {"x": round(x, 4), "y": round(y, 4)}

            radius += 0.05

        # Fallback: return last computed position
        return {"x": round(x, 4), "y": round(y, 4)}

    def signals_for_turn(
        self,
        new_stars: list,
        existing_stars: list,
    ) -> list:
        """Main entry point. Emit fog + star signals for each new star.

        Returns signals sorted: all fog_disturbance first, then all
        star_created.
        """
        fog_signals = []
        star_signals = []
        # Track positions as we place stars to avoid overlap among new ones
        all_existing = list(existing_stars)

        for star in new_stars:
            dim = star["dimension"]
            confidence = star["confidence"]
            label = star["label"]
            color = star["star_color"]

            fog = self.create_fog_disturbance(dim, confidence, all_existing)
            position = fog["position"]

            created = self.create_star_created(dim, label, color, position)

            fog_signals.append(fog)
            star_signals.append(created)

            # Add to existing so next star avoids overlap
            all_existing.append({"dimension": dim, "position": position})

        return fog_signals + star_signals


# ── Helpers ─────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash_to_angle(dimension: str) -> float:
    """Deterministic angle (radians) from dimension name."""
    h = hashlib.md5(dimension.encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2 * math.pi


def _min_distance(x: float, y: float, positions: list[tuple]) -> float:
    """Minimum Euclidean distance from (x,y) to any existing position."""
    if not positions:
        return float("inf")
    return min(
        math.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in positions
    )
