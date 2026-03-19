# Sufficiency Scorer: Insight-Driven Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the scorer from "count activated detectors" to "can we deliver 3+ reframeable insights that make the user feel understood."

**Architecture:** The scorer becomes a quality gate, not a coverage meter. Detectors run and produce `DetectorResult`s as before. A new `InsightExtractor` crosses detector results to produce `InsightCandidate`s — specific, reframeable findings about the user. The scorer counts high-quality candidates: ≥3 = ready to bloom. The orchestrator manages cumulative text across multiple presses, enforcing a 40-word minimum before running detectors.

**Tech Stack:** Python 3.12, pydantic, pytest, pytest-asyncio

**What stays:** All 11 detector adapters (`detectors/*.py`) are unchanged. `detectors/eq.py` (behavioral features) is unchanged. Parallel execution in orchestrator stays.

**What changes:** `models.py`, `config.py`, `scorer.py`, `orchestrator.py` — rewritten. New `insight_extractor.py`. All tests rewritten.

---

### Task 1: Rewrite models.py — Insight-oriented data structures

**Files:**
- Modify: `sufficiency_scorer/models.py`
- Test: `tests/test_models.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_models.py
"""Tests for insight-oriented data models."""

import pytest
from sufficiency_scorer.models import (
    Dimension,
    DetectorResult,
    InsightCandidate,
    InsightQuality,
    SufficiencyReport,
    SessionState,
)


class TestInsightCandidate:
    def test_create_valid_insight(self):
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
            signal="high frustration + avoidance style",
            reframe="You pick your battles carefully — not out of weakness, but because you know which fights matter",
            quality=InsightQuality.HIGH,
            confidence=0.75,
        )
        assert insight.quality == InsightQuality.HIGH
        assert len(insight.source_dimensions) == 2

    def test_insight_quality_ordering(self):
        assert InsightQuality.HIGH.value > InsightQuality.MEDIUM.value
        assert InsightQuality.MEDIUM.value > InsightQuality.LOW.value
        assert InsightQuality.LOW.value > InsightQuality.NOISE.value


class TestSufficiencyReport:
    def test_ready_with_3_high_quality(self):
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="strong frustration",
                reframe="You have a sharp sense of fairness",
                quality=InsightQuality.HIGH,
                confidence=0.8,
            )
            for _ in range(3)
        ]
        report = SufficiencyReport(
            ready=True,
            insights=insights,
            detector_results=[],
            ring_progress=1.0,
            prompt_hint="ready",
        )
        assert report.ready is True
        assert len(report.insights) == 3

    def test_not_ready_with_2_insights(self):
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="moderate sadness",
                reframe="You feel things deeply",
                quality=InsightQuality.HIGH,
                confidence=0.7,
            )
            for _ in range(2)
        ]
        report = SufficiencyReport(
            ready=False,
            insights=insights,
            detector_results=[],
            ring_progress=0.67,
            prompt_hint="keep_going",
        )
        assert report.ready is False

    def test_ring_progress_clamped(self):
        report = SufficiencyReport(
            ready=True,
            insights=[],
            detector_results=[],
            ring_progress=1.0,
            prompt_hint="ready",
        )
        assert 0.0 <= report.ring_progress <= 1.0


class TestSessionState:
    def test_accumulate_text(self):
        state = SessionState()
        state.add_segment("I'm really stressed about work.")
        assert state.word_count >= 5
        assert len(state.segments) == 1

    def test_multi_segment_accumulation(self):
        state = SessionState()
        state.add_segment("My boss keeps pushing me.")
        state.add_segment("I don't know what to do anymore.")
        assert len(state.segments) == 2
        assert "boss" in state.full_text
        assert "anymore" in state.full_text

    def test_word_count_tracks_total(self):
        state = SessionState()
        state.add_segment("one two three four five")
        state.add_segment("six seven eight nine ten")
        assert state.word_count == 10

    def test_meets_minimum_words(self):
        state = SessionState(min_words=40)
        state.add_segment("Short text here.")
        assert state.meets_minimum is False
        state.add_segment(
            "I have been feeling really overwhelmed lately with everything "
            "going on at work and my boss keeps asking me to do overtime "
            "and I just can't take it anymore honestly"
        )
        assert state.meets_minimum is True
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_models.py -v`
Expected: FAIL — `InsightCandidate`, `InsightQuality`, `SessionState` don't exist yet

**Step 3: Write minimal implementation**

```python
# sufficiency_scorer/models.py
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
    NOISE = 0    # generic / could apply to anyone
    LOW = 1      # somewhat specific but shallow
    MEDIUM = 2   # specific, reframeable, but single-dimension
    HIGH = 3     # cross-dimensional, deeply specific, "how did you know?"


class InsightCandidate(BaseModel):
    """A packageable finding — something we can reframe back to the user.

    This is the "AlphaGo move 37" — not what the user said, but what we
    inferred by crossing multiple detector signals.
    """
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_models.py -v`
Expected: PASS (all 9 tests)

**Step 5: Commit**

```bash
git add sufficiency_scorer/models.py tests/test_models.py
git commit -m "refactor: insight-oriented models — InsightCandidate, SessionState, new SufficiencyReport"
```

---

### Task 2: Rewrite config.py — Insight thresholds replace coverage weights

**Files:**
- Modify: `sufficiency_scorer/config.py`
- Test: (no separate test — config is just constants, tested via scorer)

**Step 1: Write the new config**

```python
# sufficiency_scorer/config.py
"""Scoring thresholds and parallelization config."""

from sufficiency_scorer.models import Dimension

# === Insight Gate ===
# Minimum high-quality insights needed to trigger bloom
INSIGHT_THRESHOLD = 3

# Minimum quality tier that counts toward the threshold
MIN_INSIGHT_QUALITY_FOR_READY = 2  # InsightQuality.MEDIUM or higher

# === Word Gate ===
# Minimum words before running detectors (cumulative across presses)
MIN_WORDS = 40

# === Confidence Thresholds ===
# Minimum detector confidence to consider for insight extraction
MIN_CONFIDENCE = 0.15

# Minimum confidence for a cross-dimensional insight to count
MIN_INSIGHT_CONFIDENCE = 0.4

# === Parallelization ===
PARALLEL_GROUPS: list[list[Dimension]] = [
    # Group 0: zero-cost, no LLM (run first, <10ms)
    [Dimension.EQ],
    # Group 1: all LLM-based detectors (run in parallel, ~1-2s)
    [
        Dimension.EMOTION,
        Dimension.CONFLICT,
        Dimension.HUMOR,
        Dimension.MBTI,
        Dimension.FRAGILITY,
        Dimension.LOVE_LANGUAGE,
        Dimension.CONNECTION_RESPONSE,
        Dimension.CHARACTER,
        Dimension.COMMUNICATION_DNA,
        Dimension.SOULGRAPH,
    ],
]
```

**Step 2: Commit**

```bash
git add sufficiency_scorer/config.py
git commit -m "refactor: config — replace coverage weights with insight thresholds"
```

---

### Task 3: Create insight_extractor.py — The "AlphaGo" layer

This is the core new module. It takes raw detector results and produces `InsightCandidate`s by crossing signals across dimensions.

**Files:**
- Create: `sufficiency_scorer/insight_extractor.py`
- Test: `tests/test_insight_extractor.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_insight_extractor.py
"""Tests for insight extraction — crossing detector signals into reframeable insights."""

import pytest
from sufficiency_scorer.models import DetectorResult, Dimension, InsightCandidate, InsightQuality
from sufficiency_scorer.insight_extractor import InsightExtractor


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0, detail: dict | None = None) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


class TestSingleDimensionInsights:
    """Each activated detector can produce at least one insight."""

    def test_emotion_produces_insight(self):
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) >= 1
        assert any(i.source_dimensions == [Dimension.EMOTION] for i in insights)

    def test_high_emotion_is_high_quality(self):
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55), ("anger", 0.52)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        emotion_insights = [i for i in insights if Dimension.EMOTION in i.source_dimensions]
        assert any(i.quality >= InsightQuality.MEDIUM for i in emotion_insights)

    def test_humor_produces_insight(self):
        results = [make_result(Dimension.HUMOR, True, 0.6, detail={
            "humor_detected": True,
            "styles": {"self_deprecating": 0.8, "affiliative": 0.3},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) >= 1

    def test_conflict_produces_insight(self):
        results = [make_result(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7, "compromise": 0.4},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) >= 1

    def test_fragility_produces_insight(self):
        results = [make_result(Dimension.FRAGILITY, True, 0.7, detail={
            "pattern": "open",
            "pattern_scores": {"open": 0.8, "defensive": 0.1},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) >= 1

    def test_eq_produces_insight(self):
        results = [make_result(Dimension.EQ, True, 0.6, detail={
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35,
            "distress": 0.43,
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) >= 1

    def test_inactive_detector_produces_nothing(self):
        results = [make_result(Dimension.HUMOR, False, 0.05)]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) == 0


class TestCrossDimensionalInsights:
    """Crossing multiple detectors produces higher-quality insights."""

    def test_emotion_plus_conflict_cross_insight(self):
        """frustration + avoidance → 'You pick your battles carefully'"""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55), ("anger", 0.52)],
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7, "compromise": 0.4},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        assert len(cross) >= 1
        assert cross[0].quality >= InsightQuality.HIGH

    def test_humor_plus_fragility_cross_insight(self):
        """self-deprecating humor + open fragility → 'You use humor to stay honest'"""
        results = [
            make_result(Dimension.HUMOR, True, 0.7, detail={
                "humor_detected": True,
                "styles": {"self_deprecating": 0.8},
            }),
            make_result(Dimension.FRAGILITY, True, 0.6, detail={
                "pattern": "open",
                "pattern_scores": {"open": 0.7},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        assert len(cross) >= 1

    def test_emotion_plus_eq_cross_insight(self):
        """high distress + high self-reference → 'You're deeply self-aware'"""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("sadness", 0.6), ("anxiety", 0.5)],
            }),
            make_result(Dimension.EQ, True, 0.6, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 30},
                "valence": -0.35,
                "distress": 0.43,
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        assert len(cross) >= 1


class TestInsightCount:
    """The number of insights drives the bloom decision."""

    def test_work_pressure_scenario_gets_3_plus(self):
        """Full work pressure scenario should produce enough for bloom."""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
            }),
            make_result(Dimension.EQ, True, 0.6, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
                "valence": -0.35,
                "distress": 0.43,
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7, "compromise": 0.4},
            }),
            make_result(Dimension.FRAGILITY, True, 0.5, detail={
                "pattern": "open",
                "pattern_scores": {"open": 0.6},
            }),
            make_result(Dimension.SOULGRAPH, True, 0.5, detail={
                "items": 2, "avg_specificity": 0.5,
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        good_insights = [i for i in insights if i.quality >= InsightQuality.MEDIUM]
        assert len(good_insights) >= 3

    def test_gibberish_gets_zero(self):
        """No activated detectors → no insights."""
        results = [make_result(dim) for dim in Dimension]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        assert len(insights) == 0

    def test_single_weak_detector_not_enough(self):
        """Only EQ faintly activated → maybe 1 insight, not 3."""
        results = [make_result(dim) for dim in Dimension]
        results = [
            r if r.dimension != Dimension.EQ
            else make_result(Dimension.EQ, True, 0.2, detail={
                "features": {"self_ref": 0.05, "question_ratio": 0.0, "words": 10},
                "valence": -0.1,
                "distress": 0.1,
            })
            for r in results
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        good_insights = [i for i in insights if i.quality >= InsightQuality.MEDIUM]
        assert len(good_insights) < 3


class TestReframeQuality:
    """Reframes must be specific, not generic platitudes."""

    def test_reframe_is_not_empty(self):
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        for insight in insights:
            assert len(insight.reframe) > 10

    def test_reframe_does_not_just_repeat_emotion_name(self):
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)
        for insight in insights:
            # Reframe should NOT be "You feel frustrated" — that's just repeating
            assert "you feel frustrated" not in insight.reframe.lower()
            assert "you are frustrated" not in insight.reframe.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_insight_extractor.py -v`
Expected: FAIL — `InsightExtractor` doesn't exist

**Step 3: Write minimal implementation**

```python
# sufficiency_scorer/insight_extractor.py
"""Insight extraction — the 'AlphaGo layer' that crosses detector signals.

Takes raw DetectorResults and produces InsightCandidates by:
1. Single-dimension insights: reframe each activated detector's top finding
2. Cross-dimensional insights: combine signals from multiple detectors
   for deeper, more surprising observations

The reframes follow the MBTI principle: take what the user told us,
say it back in a framework they didn't expect, 100% accurate but 100% fresh.
"""

from sufficiency_scorer.config import MIN_CONFIDENCE, MIN_INSIGHT_CONFIDENCE
from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)


# === Reframe Templates ===
# Each maps (detector signal pattern) → (positive reframing)
# These are rule-based: deterministic, no LLM, no randomness.

EMOTION_REFRAMES: dict[str, str] = {
    "frustration": "You have a sharp sense of when things aren't right",
    "anger": "You care deeply about fairness",
    "sadness": "You feel things at a depth most people don't reach",
    "fear": "You're tuned into risks others might miss",
    "anxiety": "Your mind is always working to protect you",
    "confusion": "You're honest enough to sit with uncertainty instead of faking clarity",
    "hope": "Even in difficulty, you hold onto what matters",
    "despair": "You've been carrying something heavy for a while",
    "agitation": "There's an energy in you that's looking for the right outlet",
    "interest": "Your curiosity runs deep",
    "amusement": "You find light even in dark places",
    "love": "Connection is central to who you are",
    "determination": "When you decide something matters, you don't let go easily",
    "guilt": "You hold yourself to a high standard",
    "shame": "You're harder on yourself than you need to be",
    "distrust": "You've learned to read people carefully",
    "grief": "The depth of your loss reflects the depth of what you had",
    "irritation": "You notice friction that others overlook",
    "regret": "You're someone who learns from the past, not just lives in it",
    "defensiveness": "You've built walls for good reasons",
}

CONFLICT_REFRAMES: dict[str, str] = {
    "avoid": "You pick your battles carefully — not out of weakness, but because you know which fights matter",
    "confront": "You face things head-on when it counts",
    "compromise": "You instinctively look for the middle ground",
    "collaborate": "You believe the best solutions come from working together",
    "compete": "When something matters to you, you fight for it",
}

HUMOR_REFRAMES: dict[str, str] = {
    "self_deprecating": "You use humor to stay honest about yourself — that takes real courage",
    "affiliative": "You naturally bring people together through laughter",
    "aggressive": "Your humor has an edge — you see through pretense",
    "self_enhancing": "You've figured out how to find lightness in heavy moments",
}

FRAGILITY_REFRAMES: dict[str, str] = {
    "open": "You have the courage to be vulnerable — that's rare",
    "defensive": "You've learned to protect yourself, and that's kept you going",
    "masked": "You hold a lot inside to keep things together for others",
    "denial": "You're stronger than you think — even if you're not ready to look at everything yet",
}

# Cross-dimensional patterns: (dim_a, signal_a, dim_b, signal_b) → reframe
CROSS_PATTERNS: list[dict] = [
    {
        "dims": [Dimension.EMOTION, Dimension.CONFLICT],
        "condition": lambda e, c: _has_emotion(e, "frustration", 0.4) and _has_style(c, "avoid", 0.5),
        "reframe": "You feel the friction but choose restraint — you're more strategic than you give yourself credit for",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.EMOTION, Dimension.CONFLICT],
        "condition": lambda e, c: _has_emotion(e, "anger", 0.4) and _has_style(c, "confront", 0.5),
        "reframe": "When you see injustice, you don't look away — that takes backbone",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.HUMOR, Dimension.FRAGILITY],
        "condition": lambda h, f: _has_humor_style(h, "self_deprecating", 0.5) and _has_fragility(f, "open"),
        "reframe": "You laugh at yourself not to hide, but to stay real — humor is your honesty tool",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.HUMOR, Dimension.FRAGILITY],
        "condition": lambda h, f: _has_humor_style(h, "self_deprecating", 0.5) and _has_fragility(f, "masked"),
        "reframe": "Your jokes carry more weight than people realize — there's something underneath worth exploring",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.EMOTION, Dimension.EQ],
        "condition": lambda e, eq: _has_any_emotion(e, 0.4) and _has_high_self_ref(eq),
        "reframe": "You're deeply self-aware — you don't just feel things, you try to understand them",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.EMOTION, Dimension.EQ],
        "condition": lambda e, eq: _has_any_emotion(e, 0.4) and _has_high_question_ratio(eq),
        "reframe": "You're not just venting — you're actively searching for answers",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.EMOTION, Dimension.FRAGILITY],
        "condition": lambda e, f: _has_emotion(e, "sadness", 0.4) and _has_fragility(f, "open"),
        "reframe": "You let yourself feel the hard things instead of running — that's braver than most people manage",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.EMOTION, Dimension.FRAGILITY],
        "condition": lambda e, f: _has_emotion(e, "anxiety", 0.4) and _has_fragility(f, "defensive"),
        "reframe": "You've built a shield, but the fact that you're here means part of you is ready to set it down",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.CONFLICT, Dimension.EQ],
        "condition": lambda c, eq: _has_style(c, "avoid", 0.5) and _has_high_self_ref(eq),
        "reframe": "You think before you act in conflict — that's emotional intelligence, not avoidance",
        "quality": InsightQuality.HIGH,
    },
    {
        "dims": [Dimension.FRAGILITY, Dimension.EQ],
        "condition": lambda f, eq: _has_fragility(f, "open") and _has_negative_valence(eq),
        "reframe": "You're going through something hard and you're not pretending otherwise — that honesty is your strength",
        "quality": InsightQuality.HIGH,
    },
]


def _has_emotion(result: DetectorResult, emotion: str, threshold: float) -> bool:
    top = result.detail.get("top_emotions", [])
    return any(name == emotion and score >= threshold for name, score in top)


def _has_any_emotion(result: DetectorResult, threshold: float) -> bool:
    top = result.detail.get("top_emotions", [])
    return any(score >= threshold for _, score in top)


def _has_style(result: DetectorResult, style: str, threshold: float) -> bool:
    styles = result.detail.get("styles", {})
    return styles.get(style, 0.0) >= threshold


def _has_humor_style(result: DetectorResult, style: str, threshold: float) -> bool:
    styles = result.detail.get("styles", {})
    return result.detail.get("humor_detected", False) and styles.get(style, 0.0) >= threshold


def _has_fragility(result: DetectorResult, pattern: str) -> bool:
    return result.detail.get("pattern") == pattern


def _has_high_self_ref(result: DetectorResult) -> bool:
    features = result.detail.get("features", {})
    return features.get("self_ref", 0.0) >= 0.08


def _has_high_question_ratio(result: DetectorResult) -> bool:
    features = result.detail.get("features", {})
    return features.get("question_ratio", 0.0) >= 0.2


def _has_negative_valence(result: DetectorResult) -> bool:
    return result.detail.get("valence", 0.0) < -0.15


class InsightExtractor:
    """Extracts reframeable insights from detector results.

    Two passes:
    1. Cross-dimensional: match patterns across pairs of detectors (HIGH quality)
    2. Single-dimensional: top signal from each activated detector (MEDIUM quality)

    Cross-dimensional insights are checked first because they're more
    surprising ("how did you know?"). Single-dimension fills the rest.
    """

    def extract(self, results: list[DetectorResult]) -> list[InsightCandidate]:
        activated = {r.dimension: r for r in results if r.activated and r.confidence >= MIN_CONFIDENCE}
        if not activated:
            return []

        insights: list[InsightCandidate] = []
        used_dims: set[tuple[Dimension, str]] = set()  # track to avoid redundancy

        # Pass 1: cross-dimensional
        for pattern in CROSS_PATTERNS:
            dims = pattern["dims"]
            if not all(d in activated for d in dims):
                continue
            detector_results = [activated[d] for d in dims]
            try:
                if pattern["condition"](*detector_results):
                    signal_parts = []
                    for d in dims:
                        signal_parts.append(f"{d.value}: {_summarize_signal(activated[d])}")
                    insights.append(InsightCandidate(
                        source_dimensions=dims,
                        signal=" + ".join(signal_parts),
                        reframe=pattern["reframe"],
                        quality=pattern["quality"],
                        confidence=min(activated[d].confidence for d in dims),
                    ))
                    for d in dims:
                        used_dims.add((d, _top_signal_key(activated[d])))
            except Exception:
                continue

        # Pass 2: single-dimensional
        for dim, result in activated.items():
            single_insights = self._extract_single(dim, result)
            for insight in single_insights:
                # Skip if this exact signal was already covered by a cross-insight
                signal_key = (dim, _top_signal_key(result))
                if signal_key in used_dims:
                    continue
                insights.append(insight)
                used_dims.add(signal_key)

        # Sort by quality descending, then confidence
        insights.sort(key=lambda i: (i.quality.value, i.confidence), reverse=True)
        return insights

    def _extract_single(self, dim: Dimension, result: DetectorResult) -> list[InsightCandidate]:
        """Extract insight(s) from a single activated detector."""
        if dim == Dimension.EMOTION:
            return self._emotion_insights(result)
        elif dim == Dimension.CONFLICT:
            return self._conflict_insights(result)
        elif dim == Dimension.HUMOR:
            return self._humor_insights(result)
        elif dim == Dimension.FRAGILITY:
            return self._fragility_insights(result)
        elif dim == Dimension.EQ:
            return self._eq_insights(result)
        elif dim == Dimension.SOULGRAPH:
            return self._soulgraph_insights(result)
        elif dim == Dimension.MBTI:
            return self._generic_insight(dim, result)
        elif dim == Dimension.COMMUNICATION_DNA:
            return self._generic_insight(dim, result)
        else:
            return []

    def _emotion_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        top_emotions = result.detail.get("top_emotions", [])
        insights = []
        for emotion_name, score in top_emotions[:2]:  # top 2 emotions
            reframe = EMOTION_REFRAMES.get(emotion_name)
            if reframe and score >= 0.3:
                insights.append(InsightCandidate(
                    source_dimensions=[Dimension.EMOTION],
                    signal=f"{emotion_name}: {score:.2f}",
                    reframe=reframe,
                    quality=InsightQuality.MEDIUM if score >= 0.4 else InsightQuality.LOW,
                    confidence=score,
                ))
        return insights

    def _conflict_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        styles = result.detail.get("styles", {})
        top_style = max(styles, key=styles.get, default=None) if styles else None
        if top_style and styles[top_style] >= 0.4:
            reframe = CONFLICT_REFRAMES.get(top_style, "")
            if reframe:
                return [InsightCandidate(
                    source_dimensions=[Dimension.CONFLICT],
                    signal=f"conflict style: {top_style} ({styles[top_style]:.2f})",
                    reframe=reframe,
                    quality=InsightQuality.MEDIUM,
                    confidence=styles[top_style],
                )]
        return []

    def _humor_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        if not result.detail.get("humor_detected", False):
            return []
        styles = result.detail.get("styles", {})
        top_style = max(styles, key=styles.get, default=None) if styles else None
        if top_style and styles[top_style] >= 0.4:
            reframe = HUMOR_REFRAMES.get(top_style, "")
            if reframe:
                return [InsightCandidate(
                    source_dimensions=[Dimension.HUMOR],
                    signal=f"humor style: {top_style} ({styles[top_style]:.2f})",
                    reframe=reframe,
                    quality=InsightQuality.MEDIUM,
                    confidence=styles[top_style],
                )]
        return []

    def _fragility_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        pattern = result.detail.get("pattern")
        if pattern:
            reframe = FRAGILITY_REFRAMES.get(pattern, "")
            if reframe:
                return [InsightCandidate(
                    source_dimensions=[Dimension.FRAGILITY],
                    signal=f"fragility pattern: {pattern}",
                    reframe=reframe,
                    quality=InsightQuality.MEDIUM,
                    confidence=result.confidence,
                )]
        return []

    def _eq_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        features = result.detail.get("features", {})
        distress = result.detail.get("distress", 0.0)
        valence = result.detail.get("valence", 0.0)
        insights = []
        if features.get("question_ratio", 0.0) >= 0.2:
            insights.append(InsightCandidate(
                source_dimensions=[Dimension.EQ],
                signal=f"high question ratio ({features['question_ratio']:.2f})",
                reframe="You're in problem-solving mode — your mind is actively looking for a way forward",
                quality=InsightQuality.MEDIUM,
                confidence=min(features["question_ratio"] * 2, 1.0),
            ))
        elif distress >= 0.3:
            insights.append(InsightCandidate(
                source_dimensions=[Dimension.EQ],
                signal=f"elevated distress ({distress:.2f})",
                reframe="What you're going through is weighing on you — and you're not ignoring it",
                quality=InsightQuality.MEDIUM,
                confidence=distress,
            ))
        elif valence < -0.2:
            insights.append(InsightCandidate(
                source_dimensions=[Dimension.EQ],
                signal=f"negative valence ({valence:.2f})",
                reframe="You're in a hard place right now, and you're being honest about it",
                quality=InsightQuality.LOW,
                confidence=abs(valence),
            ))
        return insights

    def _soulgraph_insights(self, result: DetectorResult) -> list[InsightCandidate]:
        items = result.detail.get("items", 0)
        if items >= 1:
            return [InsightCandidate(
                source_dimensions=[Dimension.SOULGRAPH],
                signal=f"extracted {items} intention items",
                reframe="There's a clear thread running through what you're saying — you know what matters to you",
                quality=InsightQuality.MEDIUM if items >= 2 else InsightQuality.LOW,
                confidence=result.confidence,
            )]
        return []

    def _generic_insight(self, dim: Dimension, result: DetectorResult) -> list[InsightCandidate]:
        """Fallback for dimensions without specialized reframe templates."""
        if result.confidence >= MIN_INSIGHT_CONFIDENCE:
            return [InsightCandidate(
                source_dimensions=[dim],
                signal=f"{dim.value} activated ({result.confidence:.2f})",
                reframe="",  # Will be filled by downstream star label generator
                quality=InsightQuality.LOW,
                confidence=result.confidence,
            )]
        return []


def _summarize_signal(result: DetectorResult) -> str:
    """One-line summary of a detector result for internal logging."""
    if result.dimension == Dimension.EMOTION:
        top = result.detail.get("top_emotions", [])[:2]
        return ", ".join(f"{n}={s:.2f}" for n, s in top) if top else "active"
    elif result.dimension == Dimension.CONFLICT:
        styles = result.detail.get("styles", {})
        top = max(styles, key=styles.get, default="unknown") if styles else "unknown"
        return f"{top}={styles.get(top, 0):.2f}"
    elif result.dimension == Dimension.EQ:
        return f"distress={result.detail.get('distress', 0):.2f}"
    return f"conf={result.confidence:.2f}"


def _top_signal_key(result: DetectorResult) -> str:
    """Unique key for the top signal of a detector, for dedup."""
    if result.dimension == Dimension.EMOTION:
        top = result.detail.get("top_emotions", [])
        return top[0][0] if top else "none"
    elif result.dimension == Dimension.CONFLICT:
        styles = result.detail.get("styles", {})
        return max(styles, key=styles.get, default="none") if styles else "none"
    elif result.dimension == Dimension.HUMOR:
        styles = result.detail.get("styles", {})
        return max(styles, key=styles.get, default="none") if styles else "none"
    elif result.dimension == Dimension.FRAGILITY:
        return result.detail.get("pattern", "none")
    return result.dimension.value
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_insight_extractor.py -v`
Expected: PASS (all ~18 tests)

**Step 5: Commit**

```bash
git add sufficiency_scorer/insight_extractor.py tests/test_insight_extractor.py
git commit -m "feat: insight extractor — cross-dimensional signal reframing (AlphaGo layer)"
```

---

### Task 4: Rewrite scorer.py — Insight-count driven, not coverage-driven

**Files:**
- Modify: `sufficiency_scorer/scorer.py`
- Rewrite: `tests/test_scorer.py`

**Step 1: Write the failing test**

```python
# tests/test_scorer.py
"""Tests for the insight-driven scoring engine."""

import pytest
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality
from sufficiency_scorer.scorer import SufficiencyScorer


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0, detail: dict | None = None) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


def all_inactive() -> list[DetectorResult]:
    return [make_result(dim) for dim in Dimension]


def work_pressure_results() -> list[DetectorResult]:
    """Simulates '我最近工作压力很大老板逼我加班我快撑不住了'"""
    base = all_inactive()
    overrides = {
        Dimension.EMOTION: make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
        }),
        Dimension.EQ: make_result(Dimension.EQ, True, 0.6, detail={
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35, "distress": 0.43,
        }),
        Dimension.CONFLICT: make_result(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7, "compromise": 0.4},
        }),
        Dimension.FRAGILITY: make_result(Dimension.FRAGILITY, True, 0.5, detail={
            "pattern": "open", "pattern_scores": {"open": 0.6},
        }),
        Dimension.SOULGRAPH: make_result(Dimension.SOULGRAPH, True, 0.5, detail={
            "items": 2, "avg_specificity": 0.5,
        }),
    }
    return [overrides.get(r.dimension, r) for r in base]


class TestReadyDecision:
    """The core question: are we ready to bloom?"""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_ready_with_rich_input(self, scorer):
        report = scorer.score(work_pressure_results())
        assert report.ready is True
        assert len(report.insights) >= 3

    def test_not_ready_with_nothing(self, scorer):
        report = scorer.score(all_inactive())
        assert report.ready is False
        assert len(report.insights) == 0

    def test_not_ready_with_single_weak_signal(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EQ
            else make_result(Dimension.EQ, True, 0.2, detail={
                "features": {"self_ref": 0.05, "question_ratio": 0.0, "words": 10},
                "valence": -0.1, "distress": 0.1,
            })
            for r in results
        ]
        report = scorer.score(results)
        assert report.ready is False


class TestRingProgress:
    """Ring progress for visual feedback — not the gate, just the animation."""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_zero_insights_zero_progress(self, scorer):
        report = scorer.score(all_inactive())
        assert report.ring_progress == 0.0

    def test_ready_means_full_progress(self, scorer):
        report = scorer.score(work_pressure_results())
        if report.ready:
            assert report.ring_progress == 1.0

    def test_partial_progress_between_0_and_1(self, scorer):
        """2 insights out of 3 needed → progress should be partial."""
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION
            else make_result(Dimension.EMOTION, True, 0.5, detail={
                "top_emotions": [("sadness", 0.5)],
            })
            for r in results
        ]
        results = [
            r if r.dimension != Dimension.EQ
            else make_result(Dimension.EQ, True, 0.5, detail={
                "features": {"self_ref": 0.1, "question_ratio": 0.0, "words": 20},
                "valence": -0.2, "distress": 0.3,
            })
            for r in results
        ]
        report = scorer.score(results)
        assert 0.0 < report.ring_progress < 1.0


class TestPromptHint:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_no_insights_tells_more(self, scorer):
        report = scorer.score(all_inactive())
        assert report.prompt_hint == "tell_me_more"

    def test_ready_says_ready(self, scorer):
        report = scorer.score(work_pressure_results())
        assert report.prompt_hint == "ready"

    def test_some_insights_keeps_going(self, scorer):
        results = all_inactive()
        results = [
            r if r.dimension != Dimension.EMOTION
            else make_result(Dimension.EMOTION, True, 0.5, detail={
                "top_emotions": [("sadness", 0.5)],
            })
            for r in results
        ]
        report = scorer.score(results)
        if not report.ready:
            assert report.prompt_hint in ("keep_going", "almost_there", "tell_me_more")


class TestInsightsPassedThrough:
    """Scorer should include the actual insight objects for downstream use."""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_insights_in_report(self, scorer):
        report = scorer.score(work_pressure_results())
        assert len(report.insights) > 0
        for insight in report.insights:
            assert len(insight.reframe) > 0 or insight.quality == InsightQuality.LOW

    def test_insights_sorted_by_quality(self, scorer):
        report = scorer.score(work_pressure_results())
        if len(report.insights) >= 2:
            for i in range(len(report.insights) - 1):
                assert report.insights[i].quality >= report.insights[i + 1].quality
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_scorer.py -v`
Expected: FAIL — scorer still has old interface

**Step 3: Write minimal implementation**

```python
# sufficiency_scorer/scorer.py
"""Core scoring engine — determines readiness based on insight quality and count."""

from sufficiency_scorer.config import INSIGHT_THRESHOLD, MIN_INSIGHT_QUALITY_FOR_READY
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import (
    DetectorResult,
    InsightQuality,
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

        # Count insights that meet quality bar
        good_insights = [
            i for i in insights
            if i.quality.value >= MIN_INSIGHT_QUALITY_FOR_READY
        ]
        good_count = len(good_insights)

        ready = good_count >= INSIGHT_THRESHOLD
        ring_progress = min(good_count / INSIGHT_THRESHOLD, 1.0) if INSIGHT_THRESHOLD > 0 else 0.0
        if ready:
            ring_progress = 1.0

        # Prompt hint
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_scorer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sufficiency_scorer/scorer.py tests/test_scorer.py
git commit -m "refactor: scorer — insight-count driven readiness, not coverage-driven"
```

---

### Task 5: Rewrite orchestrator.py — Session state + 40-word gate

**Files:**
- Modify: `sufficiency_scorer/orchestrator.py`
- Rewrite: `tests/test_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
"""Tests for the orchestrator — session accumulation, word gate, parallel execution."""

import asyncio
import pytest

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension, SessionState
from sufficiency_scorer.orchestrator import Orchestrator


class MockDetector(DetectorAdapter):
    """Mock detector that returns predetermined results."""

    def __init__(self, dimension: Dimension, activated: bool, confidence: float, detail: dict | None = None, delay: float = 0.0):
        self.dimension = dimension
        self._activated = activated
        self._confidence = confidence
        self._detail = detail or {}
        self._delay = delay
        self.call_count = 0

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return DetectorResult(
            dimension=self.dimension,
            activated=self._activated,
            confidence=self._confidence if self._activated else 0.0,
            detail=self._detail,
        )


def make_rich_mocks() -> list[MockDetector]:
    """Mocks that simulate a rich emotional input."""
    return [
        MockDetector(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55), ("anger", 0.52)],
        }),
        MockDetector(Dimension.EQ, True, 0.6, detail={
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35, "distress": 0.43,
        }),
        MockDetector(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7, "compromise": 0.4},
        }),
        MockDetector(Dimension.FRAGILITY, True, 0.5, detail={
            "pattern": "open", "pattern_scores": {"open": 0.6},
        }),
        MockDetector(Dimension.SOULGRAPH, True, 0.5, detail={
            "items": 2, "avg_specificity": 0.5,
        }),
    ] + [MockDetector(dim, False, 0.0) for dim in Dimension
         if dim not in {Dimension.EMOTION, Dimension.EQ, Dimension.CONFLICT, Dimension.FRAGILITY, Dimension.SOULGRAPH}]


class TestWordGate:
    """40-word minimum before running detectors."""

    @pytest.mark.asyncio
    async def test_under_40_words_not_ready(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        report = await orch.evaluate("Short text here.")
        assert report.ready is False
        assert report.prompt_hint == "need_more_words"

    @pytest.mark.asyncio
    async def test_under_40_words_detectors_not_called(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        await orch.evaluate("Short text.")
        for adapter in adapters:
            assert adapter.call_count == 0

    @pytest.mark.asyncio
    async def test_over_40_words_runs_detectors(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        long_text = (
            "I'm really stressed about work lately. My boss keeps pushing me "
            "to do overtime and I don't know how to say no. I feel like I'm "
            "losing myself in this job and I can't figure out what to do anymore. "
            "Every day feels the same and I'm exhausted."
        )
        report = await orch.evaluate(long_text)
        # At least some detectors should have been called
        assert any(a.call_count > 0 for a in adapters)


class TestSessionAccumulation:
    """Multi-press: text accumulates across evaluate() calls."""

    @pytest.mark.asyncio
    async def test_cumulative_text_passes_gate(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        # First press: too short
        r1 = await orch.evaluate("My boss makes me work overtime.")
        assert r1.ready is False
        # Second press: cumulative now passes 40 words
        r2 = await orch.evaluate(
            "I feel trapped and I don't know what to do. "
            "Every day is the same and I'm losing motivation. "
            "I used to love what I do but now it feels pointless."
        )
        # Now detectors should have run
        assert any(a.call_count > 0 for a in adapters)

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        adapters = make_rich_mocks()
        orch = Orchestrator(adapters=adapters)
        long_text = " ".join(["word"] * 50)
        await orch.evaluate(long_text)
        orch.reset()
        r = await orch.evaluate("Short.")
        assert r.ready is False


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_faster_than_serial(self):
        """11 detectors with 0.1s delay each — parallel should be ~0.1s, not 1.1s."""
        adapters = [MockDetector(dim, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)] if dim == Dimension.EMOTION else [],
            "styles": {"avoid": 0.7} if dim == Dimension.CONFLICT else {},
            "features": {"self_ref": 0.1, "question_ratio": 0.2, "words": 50} if dim == Dimension.EQ else {},
            "valence": -0.3 if dim == Dimension.EQ else 0.0,
            "distress": 0.4 if dim == Dimension.EQ else 0.0,
            "pattern": "open" if dim == Dimension.FRAGILITY else None,
            "humor_detected": True if dim == Dimension.HUMOR else False,
            "items": 2 if dim == Dimension.SOULGRAPH else 0,
        }, delay=0.1) for dim in Dimension]
        orch = Orchestrator(adapters=adapters)

        import time
        long_text = " ".join(["word"] * 50)
        start = time.monotonic()
        await orch.evaluate(long_text)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, f"Took {elapsed:.2f}s — detectors not running in parallel"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_failed_detector_doesnt_crash(self):
        class FailingDetector(DetectorAdapter):
            dimension = Dimension.HUMOR
            async def detect(self, text: str, **kwargs) -> DetectorResult:
                raise RuntimeError("LLM unavailable")

        adapters = [
            MockDetector(dim, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55)] if dim == Dimension.EMOTION else [],
            }) if dim != Dimension.HUMOR
            else FailingDetector()
            for dim in Dimension
        ]
        orch = Orchestrator(adapters=adapters)
        long_text = " ".join(["word"] * 50)
        report = await orch.evaluate(long_text)
        # Should not crash
        humor_result = [r for r in report.detector_results if r.dimension == Dimension.HUMOR]
        assert len(humor_result) == 1
        assert humor_result[0].activated is False
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL — orchestrator has old interface

**Step 3: Write minimal implementation**

```python
# sufficiency_scorer/orchestrator.py
"""Orchestrator — manages session state, word gate, parallel detection, and scoring."""

import asyncio

from sufficiency_scorer.config import PARALLEL_GROUPS, MIN_WORDS
from sufficiency_scorer.detectors import ALL_ADAPTERS
from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension, SessionState, SufficiencyReport
from sufficiency_scorer.scorer import SufficiencyScorer


class Orchestrator:
    """Runs the full Touch ID flow: accumulate text → check word gate → run detectors → score.

    Supports multi-press: each evaluate() call adds to the session.
    Call reset() to start a new session.
    """

    def __init__(self, adapters: list[DetectorAdapter] | None = None):
        if adapters is not None:
            self._adapters = {a.dimension: a for a in adapters}
        else:
            self._adapters = {cls.dimension: cls() for cls in ALL_ADAPTERS}
        self._scorer = SufficiencyScorer()
        self._session = SessionState(min_words=MIN_WORDS)

    def reset(self) -> None:
        """Clear session state for a new user interaction."""
        self._session = SessionState(min_words=MIN_WORDS)

    async def evaluate(self, text: str, **kwargs) -> SufficiencyReport:
        """Add text to session, check gate, run detectors if ready."""
        self._session.add_segment(text)

        if not self._session.meets_minimum:
            return SufficiencyReport(
                ready=False,
                insights=[],
                detector_results=[],
                ring_progress=0.0,
                prompt_hint="need_more_words",
            )

        full_text = self._session.full_text
        results = await self._run_detectors(full_text, **kwargs)
        return self._scorer.score(results)

    async def _run_detectors(self, text: str, **kwargs) -> list[DetectorResult]:
        """Run all detectors in parallel groups."""
        results: list[DetectorResult] = []

        for group in PARALLEL_GROUPS:
            group_adapters = [
                self._adapters[dim] for dim in group if dim in self._adapters
            ]
            if not group_adapters:
                continue
            group_results = await asyncio.gather(
                *(adapter.detect(text, **kwargs) for adapter in group_adapters),
                return_exceptions=True,
            )
            for adapter, result in zip(group_adapters, group_results):
                if isinstance(result, Exception):
                    results.append(DetectorResult(
                        dimension=adapter.dimension,
                        activated=False,
                        confidence=0.0,
                        detail={"error": str(result)},
                    ))
                else:
                    results.append(result)

        # Fill in missing dimensions
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        return results
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sufficiency_scorer/orchestrator.py tests/test_orchestrator.py
git commit -m "refactor: orchestrator — session accumulation + 40-word gate"
```

---

### Task 6: Update __init__.py and rewrite scenario tests

**Files:**
- Modify: `sufficiency_scorer/__init__.py`
- Rewrite: `tests/test_scenarios.py`
- Keep: `tests/test_eq.py` (unchanged — EQ adapter didn't change)

**Step 1: Update __init__.py**

```python
# sufficiency_scorer/__init__.py
"""SoulMap Touch ID Ring Sufficiency Scorer."""

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
from sufficiency_scorer.orchestrator import Orchestrator

__all__ = [
    "DetectorResult",
    "Dimension",
    "InsightCandidate",
    "InsightQuality",
    "InsightExtractor",
    "Orchestrator",
    "SessionState",
    "SufficiencyReport",
    "SufficiencyScorer",
]
```

**Step 2: Rewrite scenario tests**

```python
# tests/test_scenarios.py
"""Scenario validation — 9 types of user input with insight-based expectations.

Uses 10_scenarios.json from emotion-detector for real text inputs.
Tests that each scenario type produces the right number and type of insights.
"""

import json
import pytest
from pathlib import Path

from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality
from sufficiency_scorer.scorer import SufficiencyScorer
from sufficiency_scorer.detectors.eq import EQAdapter


SCENARIOS_PATH = Path.home() / "emotion-detector" / "results" / "10_scenarios.json"


def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        pytest.skip("10_scenarios.json not found")
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def make_result(dim: Dimension, activated: bool = False, confidence: float = 0.0, detail: dict | None = None) -> DetectorResult:
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


def simulate_scenario(scenario: dict, activations: dict[str, dict]) -> list[DetectorResult]:
    """Build detector results from scenario data + expected activation patterns."""
    results = []
    for dim in Dimension:
        spec = activations.get(dim.value)
        if spec and spec.get("activated"):
            results.append(make_result(dim, True, spec.get("confidence", 0.6), spec.get("detail", {})))
        else:
            results.append(make_result(dim))
    return results


# Per-scenario detector activation with detail (needed for insight extraction)
SCENARIO_DETECTORS = {
    "工作压力": {
        "emotion": {"activated": True, "confidence": 0.7, "detail": {
            "top_emotions": [("frustration", 0.55), ("anger", 0.52), ("sadness", 0.48)],
        }},
        "eq": {"activated": True, "confidence": 0.6, "detail": {
            "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
            "valence": -0.35, "distress": 0.43,
        }},
        "conflict": {"activated": True, "confidence": 0.6, "detail": {
            "styles": {"avoid": 0.7, "compromise": 0.4},
        }},
        "fragility": {"activated": True, "confidence": 0.5, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.6},
        }},
        "soulgraph": {"activated": True, "confidence": 0.5, "detail": {
            "items": 2, "avg_specificity": 0.5,
        }},
    },
    "自嘲幽默": {
        "emotion": {"activated": True, "confidence": 0.6, "detail": {
            "top_emotions": [("sadness", 0.5), ("amusement", 0.4)],
        }},
        "eq": {"activated": True, "confidence": 0.5, "detail": {
            "features": {"self_ref": 0.1, "question_ratio": 0.33, "words": 26},
            "valence": -0.15, "distress": 0.25,
        }},
        "humor": {"activated": True, "confidence": 0.7, "detail": {
            "humor_detected": True, "styles": {"self_deprecating": 0.8, "affiliative": 0.3},
        }},
        "fragility": {"activated": True, "confidence": 0.5, "detail": {
            "pattern": "masked", "pattern_scores": {"masked": 0.6},
        }},
        "soulgraph": {"activated": True, "confidence": 0.4, "detail": {
            "items": 1, "avg_specificity": 0.4,
        }},
    },
    "关系困境": {
        "emotion": {"activated": True, "confidence": 0.7, "detail": {
            "top_emotions": [("frustration", 0.6), ("sadness", 0.55), ("anger", 0.4)],
        }},
        "eq": {"activated": True, "confidence": 0.6, "detail": {
            "features": {"self_ref": 0.1, "question_ratio": 0.0, "words": 30},
            "valence": -0.4, "distress": 0.5,
        }},
        "conflict": {"activated": True, "confidence": 0.7, "detail": {
            "styles": {"avoid": 0.6, "collaborate": 0.5},
        }},
        "fragility": {"activated": True, "confidence": 0.6, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.7},
        }},
        "love_language": {"activated": True, "confidence": 0.5, "detail": {
            "has_relationship_context": True,
        }},
        "connection_response": {"activated": True, "confidence": 0.5, "detail": {
            "patterns": ["turning_away"],
        }},
        "soulgraph": {"activated": True, "confidence": 0.6, "detail": {
            "items": 3, "avg_specificity": 0.6,
        }},
    },
    "丧亲之痛": {
        "emotion": {"activated": True, "confidence": 0.8, "detail": {
            "top_emotions": [("grief", 0.8), ("sadness", 0.7), ("despair", 0.5)],
        }},
        "eq": {"activated": True, "confidence": 0.7, "detail": {
            "features": {"self_ref": 0.12, "question_ratio": 0.0, "words": 35},
            "valence": -0.6, "distress": 0.7,
        }},
        "fragility": {"activated": True, "confidence": 0.7, "detail": {
            "pattern": "open", "pattern_scores": {"open": 0.8},
        }},
        "soulgraph": {"activated": True, "confidence": 0.5, "detail": {
            "items": 2, "avg_specificity": 0.5,
        }},
    },
}


class TestScenarioInsights:
    """Each scenario type should produce enough insights to bloom (or not)."""

    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_work_pressure_ready(self, scorer):
        results = simulate_scenario({}, SCENARIO_DETECTORS["工作压力"])
        report = scorer.score(results)
        assert report.ready is True
        assert len(report.insights) >= 3

    def test_self_deprecating_humor_ready(self, scorer):
        results = simulate_scenario({}, SCENARIO_DETECTORS["自嘲幽默"])
        report = scorer.score(results)
        assert report.ready is True
        # Should include humor-related insight
        humor_insights = [i for i in report.insights if Dimension.HUMOR in i.source_dimensions]
        assert len(humor_insights) >= 1

    def test_relationship_crisis_ready_with_many(self, scorer):
        results = simulate_scenario({}, SCENARIO_DETECTORS["关系困境"])
        report = scorer.score(results)
        assert report.ready is True
        assert len(report.insights) >= 4  # richest scenario

    def test_grief_ready(self, scorer):
        results = simulate_scenario({}, SCENARIO_DETECTORS["丧亲之痛"])
        report = scorer.score(results)
        assert report.ready is True
        # Should have grief-specific insight
        assert any("grief" in i.signal or "loss" in i.reframe.lower() or "depth" in i.reframe.lower()
                    for i in report.insights)


class TestGibberishRejection:
    @pytest.fixture
    def scorer(self):
        return SufficiencyScorer()

    def test_nothing_activated_not_ready(self, scorer):
        results = [make_result(dim) for dim in Dimension]
        report = scorer.score(results)
        assert report.ready is False
        assert len(report.insights) == 0
        assert report.prompt_hint == "tell_me_more"


class TestEQOnRealScenarios:
    """Run the real EQ adapter on scenario texts — still valid."""

    @pytest.fixture
    def scenarios(self):
        return load_scenarios()

    @pytest.mark.asyncio
    async def test_work_pressure_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "工作压力")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True

    @pytest.mark.asyncio
    async def test_excited_discovery_eq(self, scenarios):
        scenario = next(s for s in scenarios if s["type"] == "兴奋发现")
        adapter = EQAdapter()
        result = await adapter.detect(scenario["text"])
        assert result.activated is True
```

**Step 3: Run all tests**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add sufficiency_scorer/__init__.py tests/test_scenarios.py
git commit -m "refactor: scenario tests — validate insight quality per scenario type"
```

---

### Task 7: Delete test_models.py leftovers from old RingSegment, final cleanup

**Files:**
- Remove old `RingSegment` references if any tests still import it
- Run full test suite

**Step 1: Run full test suite**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Final commit**

```bash
git add -A
git commit -m "refactor complete: insight-driven sufficiency scorer

Scoring shifted from 'count activated detectors' to 'can we deliver 3+
reframeable insights'. Key changes:
- InsightExtractor: cross-dimensional signal reframing (AlphaGo layer)
- Scorer: insight-count driven readiness
- Orchestrator: session accumulation + 40-word gate
- Config: insight thresholds replace coverage weights
- All tests rewritten for insight-oriented validation"
```

---

## Summary of Changes

| File | Action | What Changed |
|------|--------|-------------|
| `models.py` | Rewrite | +`InsightCandidate`, +`InsightQuality`, +`SessionState`, -`RingSegment`, new `SufficiencyReport` |
| `config.py` | Rewrite | -`WEIGHTS`, -`ACTIVATION_TARGET`, +`INSIGHT_THRESHOLD=3`, +`MIN_WORDS=40` |
| `insight_extractor.py` | **New** | Cross-dimensional reframing engine with 10 cross-patterns + per-dimension templates |
| `scorer.py` | Rewrite | From weighted coverage → insight count ≥ 3 |
| `orchestrator.py` | Rewrite | +`SessionState` accumulation, +40-word gate, +`reset()` |
| `__init__.py` | Update | New exports |
| `tests/test_models.py` | **New** | SessionState, InsightCandidate tests |
| `tests/test_insight_extractor.py` | **New** | 18 tests: single-dim, cross-dim, count, quality |
| `tests/test_scorer.py` | Rewrite | Insight-driven readiness tests |
| `tests/test_orchestrator.py` | Rewrite | Word gate, session accumulation, parallel, error handling |
| `tests/test_scenarios.py` | Rewrite | Per-scenario insight validation |
| `tests/test_eq.py` | **Unchanged** | Still valid |
| `detectors/*.py` | **Unchanged** | All 11 adapters unchanged |
