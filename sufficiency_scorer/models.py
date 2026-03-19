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
    EQ = "eq"
    FRAGILITY = "fragility"
    CONNECTION_RESPONSE = "connection_response"
    CHARACTER = "character"
    COMMUNICATION_DNA = "communication_dna"
    SOULGRAPH = "soulgraph"


class DetectorResult(BaseModel):
    """Normalized output from any detector."""
    dimension: Dimension
    activated: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: dict = Field(default_factory=dict)



class InsightQuality(int, Enum):
    """Quality tier of an insight candidate."""
    NOISE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class InsightCandidate(BaseModel):
    """A packageable finding — something we can reframe back to the user."""
    source_dimensions: list[Dimension]
    signal: str = Field(description="What the detectors found (internal)")
    reframe: str = Field(description="How to say it back to the user (positive reframing)")
    quality: InsightQuality = InsightQuality.NOISE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SufficiencyReport(BaseModel):
    """Output of the scorer — drives ring UI and star bloom."""
    ready: bool = Field(default=False, description="True = bloom, False = keep going")
    insights: list[InsightCandidate] = Field(default_factory=list)
    detector_results: list[DetectorResult] = Field(default_factory=list)
    ring_progress: float = Field(default=0.0, ge=0.0, le=1.0, description="For ring animation")
    prompt_hint: str = Field(default="", description="UI hint for what to show")



class SessionState:
    """Accumulates text across multiple presses (Touch ID multi-tap)."""

    def __init__(self, min_words: int = 40):
        self.segments: list[str] = []
        self.min_words = min_words

    def add_segment(self, text: str) -> None:
        self.segments.append(text)

    @property
    def full_text(self) -> str:
        return " ".join(self.segments)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    @property
    def meets_minimum(self) -> bool:
        return self.word_count >= self.min_words
