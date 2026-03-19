"""Detector adapters — normalize each detector's output to DetectorResult."""

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.detectors.emotion import EmotionAdapter
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.detectors.conflict import ConflictAdapter
from sufficiency_scorer.detectors.humor import HumorAdapter
from sufficiency_scorer.detectors.mbti import MBTIAdapter
from sufficiency_scorer.detectors.fragility import FragilityAdapter
from sufficiency_scorer.detectors.love_language import LoveLanguageAdapter
from sufficiency_scorer.detectors.connection_response import ConnectionResponseAdapter
from sufficiency_scorer.detectors.character import CharacterAdapter
from sufficiency_scorer.detectors.communication_dna import CommunicationDNAAdapter
from sufficiency_scorer.detectors.soulgraph import SoulGraphAdapter

ALL_ADAPTERS: list[type[DetectorAdapter]] = [
    EmotionAdapter,
    EQAdapter,
    ConflictAdapter,
    HumorAdapter,
    MBTIAdapter,
    FragilityAdapter,
    LoveLanguageAdapter,
    ConnectionResponseAdapter,
    CharacterAdapter,
    CommunicationDNAAdapter,
    SoulGraphAdapter,
]

__all__ = [
    "DetectorAdapter",
    "ALL_ADAPTERS",
    "EmotionAdapter",
    "EQAdapter",
    "ConflictAdapter",
    "HumorAdapter",
    "MBTIAdapter",
    "FragilityAdapter",
    "LoveLanguageAdapter",
    "ConnectionResponseAdapter",
    "CharacterAdapter",
    "CommunicationDNAAdapter",
    "SoulGraphAdapter",
]
