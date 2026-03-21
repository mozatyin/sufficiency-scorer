"""SoulMap Star Engine — Touch ID ring + star generation + insight packaging.

Pipeline:
  1. Orchestrator: accumulate text, 40-word gate, run detectors
  2. SufficiencyScorer: bloom gate (3+ extractable insights, zero LLM cost)
  3. InsightGenerator: personalized insights for user (1x LLM call)
  4. StarEngine: star creation, labels, fog events, safety gate
"""

from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
    SessionState,
    SufficiencyReport,
)
from sufficiency_scorer.scorer import SufficiencyScorer
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.insight_generator import InsightGenerator
from sufficiency_scorer.star_engine import StarEngine, Star, FogEvent, StarCreatedEvent
from sufficiency_scorer.orchestrator import Orchestrator
from sufficiency_scorer.precompute import precompute

__all__ = [
    "DetectorResult",
    "Dimension",
    "FogEvent",
    "InsightCandidate",
    "InsightExtractor",
    "InsightGenerator",
    "InsightQuality",
    "Orchestrator",
    "SessionState",
    "Star",
    "StarCreatedEvent",
    "StarEngine",
    "SufficiencyReport",
    "SufficiencyScorer",
    "precompute",
]
