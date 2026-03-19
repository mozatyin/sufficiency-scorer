"""Core data models for the sufficiency scorer."""

from enum import Enum
from pydantic import BaseModel, Field


class Dimension(str, Enum):
    """The 11 detection dimensions that drive the ring."""
    EMOTION = "emotion"
    CONFLICT = "conflict"
    HUMOR = "humor"
    MBTI = "mbti"
    LOVE_LANGUAGE = "love_language"
    EQ = "eq"  # behavioral features + valence + distress
    FRAGILITY = "fragility"
    CONNECTION_RESPONSE = "connection_response"
    CHARACTER = "character"
    COMMUNICATION_DNA = "communication_dna"
    SOULGRAPH = "soulgraph"  # intention extraction


class DetectorResult(BaseModel):
    """Normalized output from any detector."""
    dimension: Dimension
    activated: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: dict = Field(default_factory=dict)


class RingSegment(BaseModel):
    """One segment of the ring visualization."""
    dimension: Dimension
    filled: bool = False
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)


class SufficiencyReport(BaseModel):
    """Full output of the scorer - drives the ring UI."""
    score: float = Field(ge=0.0, le=1.0, description="0.0-1.0 ring progress")
    ready: bool = Field(default=False, description="True when ring is full, trigger star bloom")
    activated_count: int = Field(default=0, description="How many dimensions lit up")
    total_dimensions: int = Field(default=11)
    segments: list[RingSegment] = Field(default_factory=list)
    detector_results: list[DetectorResult] = Field(default_factory=list)
    prompt_hint: str = Field(default="", description="UI hint: empty=keep going, 'ready'=bloom")

    @property
    def activation_ratio(self) -> float:
        return self.activated_count / self.total_dimensions
