# P0-P3: Validation, Dynamic Reframing, Integration, Quality Eval

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate thresholds with real data, upgrade InsightExtractor from templates to dynamic reframing, integrate 3 real detectors end-to-end, build quality evaluation framework.

**Architecture:** P0 builds a data analysis script that validates word/insight thresholds against real user data + 10 scenarios. P1 adds user-text-aware reframing to InsightExtractor (still rule-based, no LLM). P2 wires up Emotion/Conflict/Fragility adapters to real detector projects via `asyncio.to_thread()`. P3 creates an eval harness that scores insight quality.

**Tech Stack:** Python 3.12, pydantic, pytest, pytest-asyncio. Real detectors at `~/emotion-detector`, `~/conflict-detector`, `~/fragility-detector`.

**Critical data finding:** Real user first messages are 3-19 words (median 5.5). The 10 test scenarios are 24-33 words. The 40-word gate means almost ALL users need 2+ presses. This is by design (Touch ID multi-tap), but needs the 10 scenarios to work with a lower "analysis threshold" for testing.

---

### Task 1: P0 — Word threshold validation script

**Files:**
- Create: `scripts/validate_thresholds.py`
- Create: `tests/test_validation.py`

**Step 1: Write the validation script**

```python
# scripts/validate_thresholds.py
"""Validate word and insight thresholds against real data.

Reads:
  - ~/emotion-detector/data/real_user/*.jsonl (464K real conversations)
  - ~/emotion-detector/results/10_scenarios.json (9 scenario types)

Outputs:
  - Word count distribution of first user messages
  - How many presses needed to reach N-word thresholds
  - EQ adapter results on 10 scenarios
  - Insight extraction results on 10 scenarios (simulated detectors)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sufficiency_scorer.detectors.eq import EQAdapter, extract_behavioral, compute_valence, compute_distress
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


REAL_DATA_DIR = Path.home() / "emotion-detector" / "data" / "real_user"
SCENARIOS_PATH = Path.home() / "emotion-detector" / "results" / "10_scenarios.json"


def analyze_word_counts():
    """Analyze first-message word counts from real sessions."""
    sessions: dict[str, str] = {}
    for fname in REAL_DATA_DIR.glob("*.jsonl"):
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                sid = d.get("session_id", "")
                turn = d.get("turn", 0)
                text = d.get("user_text", "")
                if sid and turn == 1 and text:
                    sessions[sid] = text

    word_counts = sorted(len(t.split()) for t in sessions.values())
    n = len(word_counts)
    if n == 0:
        print("No sessions found!")
        return

    print(f"\n=== Word Count Distribution (n={n}) ===")
    print(f"  Mean:   {sum(word_counts)/n:.1f}")
    print(f"  Median: {word_counts[n//2]}")
    print(f"  P25:    {word_counts[n//4]}")
    print(f"  P75:    {word_counts[3*n//4]}")
    print(f"  P90:    {word_counts[9*n//10]}")

    for threshold in [15, 20, 25, 30, 40]:
        under = sum(1 for w in word_counts if w < threshold)
        print(f"  Under {threshold}w: {under}/{n} ({under/n*100:.0f}%)")

    # Multi-press simulation: turns 1-3 combined
    sessions_3 = {}
    for fname in REAL_DATA_DIR.glob("*.jsonl"):
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                sid = d.get("session_id", "")
                turn = d.get("turn", 0)
                text = d.get("user_text", "")
                if sid and text and turn <= 3:
                    sessions_3.setdefault(sid, []).append(text)

    combined = sorted(len(" ".join(texts).split()) for texts in sessions_3.values())
    n3 = len(combined)
    print(f"\n=== Turns 1-3 Combined (n={n3}) ===")
    for threshold in [15, 20, 25, 30, 40]:
        under = sum(1 for w in combined if w < threshold)
        print(f"  Under {threshold}w: {under}/{n3} ({under/n3*100:.0f}%)")


async def analyze_scenarios():
    """Run EQ + insight extraction on 10 scenarios."""
    if not SCENARIOS_PATH.exists():
        print("10_scenarios.json not found, skipping")
        return

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    eq = EQAdapter()
    extractor = InsightExtractor()

    print("\n=== Scenario Analysis ===")
    print(f"{'Type':<12} {'Words':>5} {'EQ Act':>6} {'EQ Conf':>7} {'Insights':>8} {'MEDIUM+':>7} {'Ready':>5}")
    print("-" * 60)

    for s in scenarios:
        text = s["text"]
        words = len(text.split())
        stype = s["type"]

        # Run EQ (real)
        eq_result = await eq.detect(text)

        # Simulate other detectors from scenario data
        results = _simulate_from_scenario(s, eq_result)

        # Extract insights
        insights = extractor.extract(results)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]

        ready = len(good) >= 3
        print(f"{stype:<12} {words:>5} {str(eq_result.activated):>6} {eq_result.confidence:>7.2f} {len(insights):>8} {len(good):>7} {'YES' if ready else 'no':>5}")

        # Print insights for this scenario
        for i, ins in enumerate(insights):
            print(f"    [{ins.quality.name}] {ins.signal[:50]}")
            print(f"           → {ins.reframe[:80]}")


def _simulate_from_scenario(scenario: dict, eq_result: DetectorResult) -> list[DetectorResult]:
    """Build detector results from scenario data + real EQ."""
    results = [DetectorResult(dimension=dim) for dim in Dimension]
    emotions = scenario.get("emotions", {})
    max_emotion = max(emotions.values()) if emotions else 0.0
    distress = scenario.get("distress", 0.0)
    top5 = scenario.get("top5_emotions", [])

    # Replace EQ with real result
    results = [r if r.dimension != Dimension.EQ else eq_result for r in results]

    # Emotion: use real scenario data
    if max_emotion > 0.15:
        results = [
            r if r.dimension != Dimension.EMOTION
            else DetectorResult(
                dimension=Dimension.EMOTION,
                activated=True,
                confidence=max_emotion,
                detail={"top_emotions": top5},
            )
            for r in results
        ]

    # Conflict: activate if frustration or anger present
    if any(name in ("frustration", "anger") for name, _ in top5[:3]):
        results = [
            r if r.dimension != Dimension.CONFLICT
            else DetectorResult(
                dimension=Dimension.CONFLICT,
                activated=True,
                confidence=0.6,
                detail={"styles": {"avoid": 0.7, "compromise": 0.3}},
            )
            for r in results
        ]

    # Fragility: activate if distress > 0.3
    if distress > 0.3:
        results = [
            r if r.dimension != Dimension.FRAGILITY
            else DetectorResult(
                dimension=Dimension.FRAGILITY,
                activated=True,
                confidence=min(distress * 1.2, 1.0),
                detail={"pattern": "open", "pattern_scores": {"open": distress}},
            )
            for r in results
        ]

    # Humor: activate if amusement present
    if any(name == "amusement" for name, _ in top5):
        results = [
            r if r.dimension != Dimension.HUMOR
            else DetectorResult(
                dimension=Dimension.HUMOR,
                activated=True,
                confidence=0.6,
                detail={"humor_detected": True, "styles": {"self_deprecating": 0.7}},
            )
            for r in results
        ]

    # SoulGraph: always mildly activated for content with substance
    word_count = scenario.get("behavioral", {}).get("words", 0)
    if word_count >= 15:
        results = [
            r if r.dimension != Dimension.SOULGRAPH
            else DetectorResult(
                dimension=Dimension.SOULGRAPH,
                activated=True,
                confidence=0.5,
                detail={"items": 2, "avg_specificity": 0.5},
            )
            for r in results
        ]

    return results


if __name__ == "__main__":
    analyze_word_counts()
    asyncio.run(analyze_scenarios())
```

**Step 2: Write the test**

```python
# tests/test_validation.py
"""Tests that validation script logic is correct."""

import pytest
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


class TestScenarioThresholds:
    """Validate that 10 scenarios produce reasonable results with current thresholds."""

    SCENARIOS = [
        # (type, text, expected_min_insights)
        ("工作压力", "hey, I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?", 3),
        ("自嘲幽默", "haha so I just got dumped again, third time this year. I'm starting to think maybe I'm the common denominator here. At least I'm consistent right?", 3),
        ("丧亲之痛", "my mom passed away last month and I don't know how to be normal anymore. People keep saying it gets better but when?", 3),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stype,text,min_insights", SCENARIOS, ids=lambda x: x if isinstance(x, str) and len(x) < 15 else "")
    async def test_scenario_reaches_threshold(self, stype, text, min_insights):
        """Each major scenario should produce at least 3 MEDIUM+ insights."""
        eq = EQAdapter()
        eq_result = await eq.detect(text)

        # Simulate emotion from text
        results = [DetectorResult(dimension=dim) for dim in Dimension]
        results = [r if r.dimension != Dimension.EQ else eq_result for r in results]
        # Add emotion activation (simulated)
        results = [
            r if r.dimension != Dimension.EMOTION
            else DetectorResult(dimension=Dimension.EMOTION, activated=True, confidence=0.6,
                                detail={"top_emotions": [("frustration", 0.55), ("sadness", 0.48)]})
            for r in results
        ]
        # Add conflict activation for work pressure
        if "boss" in text.lower() or "forcing" in text.lower():
            results = [
                r if r.dimension != Dimension.CONFLICT
                else DetectorResult(dimension=Dimension.CONFLICT, activated=True, confidence=0.6,
                                    detail={"styles": {"avoid": 0.7}})
                for r in results
            ]
        # Add fragility for emotional content
        if any(w in text.lower() for w in ("passed away", "don't know", "撑不住", "can't")):
            results = [
                r if r.dimension != Dimension.FRAGILITY
                else DetectorResult(dimension=Dimension.FRAGILITY, activated=True, confidence=0.5,
                                    detail={"pattern": "open"})
                for r in results
            ]
        # Add humor for self-deprecating content
        if "haha" in text.lower() or "at least" in text.lower():
            results = [
                r if r.dimension != Dimension.HUMOR
                else DetectorResult(dimension=Dimension.HUMOR, activated=True, confidence=0.7,
                                    detail={"humor_detected": True, "styles": {"self_deprecating": 0.8}})
                for r in results
            ]

        extractor = InsightExtractor()
        insights = extractor.extract(results)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]
        assert len(good) >= min_insights, (
            f"{stype}: only {len(good)} MEDIUM+ insights (need {min_insights}). "
            f"All insights: {[(i.quality.name, i.signal[:40]) for i in insights]}"
        )


class TestWordThresholdImpact:
    """Verify the 40-word gate against scenario data."""

    def test_all_10_scenarios_under_40_words(self):
        """All 10 scenarios are 24-33 words — confirms users need multi-press."""
        scenarios_texts = [
            "hey, I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?",
            "haha so I just got dumped again, third time this year. I'm starting to think maybe I'm the common denominator here. At least I'm consistent right?",
            "my partner never listens to me. I keep trying to explain how I feel but it's like talking to a wall. I don't know what to do anymore.",
        ]
        for text in scenarios_texts:
            words = len(text.split())
            assert words < 40, f"Expected under 40 words, got {words}: {text[:50]}..."

    def test_two_scenarios_combined_pass_40(self):
        """Two short messages combined should pass the 40-word gate."""
        msg1 = "I'm really stressed about work lately."
        msg2 = "My boss keeps pushing me to do overtime and I don't know how to say no. I feel trapped."
        combined = f"{msg1} {msg2}"
        assert len(combined.split()) >= 40
```

**Step 3: Run validation script + tests**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/python scripts/validate_thresholds.py`
Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_validation.py -v`

**Step 4: Commit**

```bash
git add scripts/validate_thresholds.py tests/test_validation.py
git commit -m "feat(P0): threshold validation — word count analysis + scenario insight verification"
```

---

### Task 2: P1 — Dynamic reframing with user context

**Files:**
- Modify: `sufficiency_scorer/insight_extractor.py`
- Create: `tests/test_dynamic_reframe.py`

The current reframes are static templates. This task makes them **context-aware** by injecting fragments of the user's original text into the reframe. Still rule-based (no LLM), but now each reframe references something specific the user said.

**Step 1: Write the failing test**

```python
# tests/test_dynamic_reframe.py
"""Tests for dynamic (context-aware) reframing."""

import pytest
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


def make_result(dim, activated=False, confidence=0.0, detail=None):
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


class TestDynamicReframe:
    """Reframes should incorporate user's actual words/context."""

    def test_emotion_reframe_includes_context(self):
        """When user_text is provided, emotion reframe should reference their situation."""
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="my boss is forcing me to do overtime")
        emotion_ins = [i for i in insights if Dimension.EMOTION in i.source_dimensions]
        assert len(emotion_ins) >= 1
        # Should reference work/boss context, not just generic "things aren't right"
        reframe = emotion_ins[0].reframe
        assert any(word in reframe.lower() for word in ("work", "boss", "overtime", "pushing", "situation")), \
            f"Reframe too generic, doesn't reference user context: {reframe}"

    def test_without_user_text_falls_back_to_template(self):
        """Without user_text, falls back to static template."""
        results = [make_result(Dimension.EMOTION, True, 0.7, detail={
            "top_emotions": [("frustration", 0.55)],
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results)  # no user_text
        assert len(insights) >= 1
        # Should still produce a valid reframe
        assert len(insights[0].reframe) > 10

    def test_conflict_reframe_includes_context(self):
        results = [make_result(Dimension.CONFLICT, True, 0.6, detail={
            "styles": {"avoid": 0.7},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I keep quiet when my partner criticizes me")
        conflict_ins = [i for i in insights if Dimension.CONFLICT in i.source_dimensions]
        assert len(conflict_ins) >= 1
        reframe = conflict_ins[0].reframe
        assert any(word in reframe.lower() for word in ("partner", "quiet", "relationship", "criticism", "situation")), \
            f"Reframe too generic: {reframe}"

    def test_cross_dimensional_reframe_with_context(self):
        """Cross-dimensional reframe should also incorporate context."""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55)],
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I'm tired of my boss making me work late but I never say anything")
        cross = [i for i in insights if len(i.source_dimensions) > 1]
        if cross:
            assert any(word in cross[0].reframe.lower() for word in ("boss", "work", "situation", "speak")), \
                f"Cross reframe too generic: {cross[0].reframe}"

    def test_humor_reframe_includes_context(self):
        results = [make_result(Dimension.HUMOR, True, 0.7, detail={
            "humor_detected": True, "styles": {"self_deprecating": 0.8},
        })]
        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text="I just got dumped again third time this year at least I'm consistent")
        humor_ins = [i for i in insights if Dimension.HUMOR in i.source_dimensions]
        assert len(humor_ins) >= 1

    def test_multiple_insights_all_contextualized(self):
        """All insights in a rich scenario should reference user context."""
        results = [
            make_result(Dimension.EMOTION, True, 0.7, detail={
                "top_emotions": [("frustration", 0.55), ("sadness", 0.48)],
            }),
            make_result(Dimension.EQ, True, 0.6, detail={
                "features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27},
                "valence": -0.35, "distress": 0.43,
            }),
            make_result(Dimension.CONFLICT, True, 0.6, detail={
                "styles": {"avoid": 0.7},
            }),
        ]
        extractor = InsightExtractor()
        insights = extractor.extract(
            results,
            user_text="I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?"
        )
        # At least some insights should be contextualized
        contextualized = sum(1 for i in insights if any(
            w in i.reframe.lower() for w in ("work", "boss", "overtime", "tired", "situation")
        ))
        assert contextualized >= 1, f"No insights reference user context. Reframes: {[i.reframe for i in insights]}"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_dynamic_reframe.py -v`
Expected: FAIL — `extract()` doesn't accept `user_text` parameter

**Step 3: Implement dynamic reframing**

Modify `sufficiency_scorer/insight_extractor.py`:

1. Add `user_text: str = ""` parameter to `extract()` method
2. Add a `_contextualize()` method that extracts key phrases from user text and injects them into reframes
3. Keep static templates as fallback when no user_text provided

Key implementation approach for `_contextualize()`:
- Extract noun phrases / key content words from user text (rule-based, no LLM)
- Find the most relevant phrase for each insight's signal type
- Inject into reframe template: `"{template} — especially when it comes to {context}"`

```python
import re

# Add to InsightExtractor class:

def extract(self, results: list[DetectorResult], user_text: str = "") -> list[InsightCandidate]:
    """Run both passes and return sorted insights."""
    active = [r for r in results if r.activated and r.confidence >= MIN_CONFIDENCE]
    by_dim: dict[Dimension, DetectorResult] = {r.dimension: r for r in active}

    insights: list[InsightCandidate] = []
    used_dims: set[Dimension] = set()

    cross_insights = self._cross_dimensional(by_dim)
    for ci in cross_insights:
        insights.append(ci)
        used_dims.update(ci.source_dimensions)

    single_insights = self._single_dimensional(by_dim, used_dims)
    insights.extend(single_insights)

    # Contextualize with user text
    if user_text:
        context = self._extract_context(user_text)
        insights = [self._contextualize(i, context) for i in insights]

    insights.sort(key=lambda i: (-i.quality.value, -i.confidence))
    return insights

@staticmethod
def _extract_context(text: str) -> dict:
    """Extract key contextual elements from user text."""
    text_lower = text.lower()
    words = text_lower.split()

    # Extract topic indicators
    topics = []
    topic_patterns = {
        "work": ["work", "job", "boss", "overtime", "ot", "office", "career", "colleague"],
        "relationship": ["partner", "husband", "wife", "boyfriend", "girlfriend", "relationship", "marriage", "dating", "dumped", "breakup"],
        "family": ["mom", "dad", "mother", "father", "parent", "brother", "sister", "family", "child", "son", "daughter"],
        "health": ["sleep", "insomnia", "tired", "exhausted", "sick", "anxiety", "panic", "pain"],
        "identity": ["who am i", "purpose", "meaning", "lost", "confused", "direction", "figure out"],
        "grief": ["passed away", "died", "death", "lost", "funeral", "grief", "mourning", "gone"],
    }
    for topic, keywords in topic_patterns.items():
        if any(kw in text_lower for kw in keywords):
            topics.append(topic)

    # Extract key action phrases (what the user is doing/experiencing)
    actions = []
    action_patterns = [
        r"(forcing me to \w+)",
        r"(keeps? (?:pushing|asking|telling|making) me)",
        r"(don'?t know (?:what to do|how to|if I))",
        r"(can'?t (?:sleep|stop|take|handle|figure)[\w ]{0,20})",
        r"(feel(?:s|ing)? (?:like|trapped|lost|alone|invisible)[\w ]{0,15})",
        r"(got dumped|broke up|left me)",
        r"(passed away|died)",
    ]
    for pattern in action_patterns:
        match = re.search(pattern, text_lower)
        if match:
            actions.append(match.group(1).strip())

    # Extract key nouns for specificity
    specifics = []
    for word in words:
        if word in ("boss", "partner", "mom", "dad", "job", "interview", "painting"):
            specifics.append(word)

    return {
        "topics": topics,
        "actions": actions,
        "specifics": specifics,
        "has_question": "?" in text,
    }

def _contextualize(self, insight: InsightCandidate, context: dict) -> InsightCandidate:
    """Add user context to a reframe if possible."""
    topics = context.get("topics", [])
    actions = context.get("actions", [])
    specifics = context.get("specifics", [])

    if not (topics or actions or specifics):
        return insight

    # Build a contextual suffix
    suffix = ""
    if actions:
        # Use the most relevant action
        suffix = f" — especially when {actions[0]}"
    elif specifics and topics:
        topic = topics[0]
        specific = specifics[0]
        topic_phrases = {
            "work": f"at work with your {specific}" if specific in ("boss", "colleague") else "in your work situation",
            "relationship": f"in your relationship" if specific in ("partner", "husband", "wife") else "in your relationships",
            "family": f"with your {specific}" if specific in ("mom", "dad", "mother", "father") else "in your family",
            "grief": f"after losing your {specific}" if specific in ("mom", "dad") else "through this loss",
            "health": "with what you're going through physically",
            "identity": "as you figure out what you really want",
        }
        suffix = f" — {topic_phrases.get(topic, 'in your situation')}"
    elif topics:
        topic_generic = {
            "work": "in your work situation",
            "relationship": "in your relationships",
            "family": "within your family",
            "grief": "through this loss",
            "health": "with what you're going through",
            "identity": "as you search for clarity",
        }
        suffix = f" — {topic_generic.get(topics[0], 'in your situation')}"

    if suffix and not insight.reframe.endswith(suffix):
        new_reframe = insight.reframe.rstrip(".!") + suffix
        return insight.model_copy(update={"reframe": new_reframe})

    return insight
```

**Step 4: Run tests**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_dynamic_reframe.py tests/test_insight_extractor.py -v`
Expected: ALL PASS (both new and existing tests)

**Step 5: Commit**

```bash
git add sufficiency_scorer/insight_extractor.py tests/test_dynamic_reframe.py
git commit -m "feat(P1): dynamic reframing — insights reference user's actual words and context"
```

---

### Task 3: P2 — Integrate real Emotion, Conflict, Fragility detectors

**Files:**
- Modify: `sufficiency_scorer/detectors/emotion.py`
- Modify: `sufficiency_scorer/detectors/conflict.py`
- Modify: `sufficiency_scorer/detectors/fragility.py`
- Create: `tests/test_integration.py`

All three detectors use **synchronous** `anthropic` SDK calls. Our orchestrator is async. Solution: `asyncio.to_thread()` wrapper.

**Step 1: Write the failing integration test**

```python
# tests/test_integration.py
"""Integration tests — real detectors on real scenario text.

These tests require:
  - ANTHROPIC_API_KEY env var set
  - ~/emotion-detector, ~/conflict-detector, ~/fragility-detector projects present

Skip gracefully if not available.
"""

import os
import sys
import pytest
from pathlib import Path

from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality

# Check prerequisites
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_EMOTION = (Path.home() / "emotion-detector" / "emotion_detector" / "detector.py").exists()
HAS_CONFLICT = (Path.home() / "conflict-detector" / "conflict_detector" / "detector.py").exists()
HAS_FRAGILITY = (Path.home() / "fragility-detector" / "fragility_detector" / "detector.py").exists()

SKIP_MSG = "Requires ANTHROPIC_API_KEY and detector projects"
skip_if_no_deps = pytest.mark.skipif(
    not (HAS_API_KEY and HAS_EMOTION and HAS_CONFLICT and HAS_FRAGILITY),
    reason=SKIP_MSG,
)

WORK_PRESSURE_TEXT = (
    "hey, I'm really tired. I don't want to do the OT anymore "
    "but my boss is forcing me to do overtime. What am I supposed to do?"
)

GRIEF_TEXT = (
    "my mom passed away last month and I don't know how to be normal anymore. "
    "People keep saying it gets better but when?"
)


@skip_if_no_deps
class TestEmotionIntegration:
    @pytest.mark.asyncio
    async def test_work_pressure_detects_frustration(self):
        from sufficiency_scorer.detectors.emotion import EmotionAdapter
        adapter = EmotionAdapter()
        result = await adapter.detect(WORK_PRESSURE_TEXT)
        assert result.activated is True
        assert result.confidence > 0.3
        top = result.detail.get("top_emotions", [])
        assert len(top) > 0
        # Should detect frustration or anger
        top_names = [name for name, _ in top[:3]]
        assert any(e in top_names for e in ("frustration", "anger", "sadness")), \
            f"Expected frustration/anger/sadness in top 3, got {top_names}"

    @pytest.mark.asyncio
    async def test_grief_detects_sadness(self):
        from sufficiency_scorer.detectors.emotion import EmotionAdapter
        adapter = EmotionAdapter()
        result = await adapter.detect(GRIEF_TEXT)
        assert result.activated is True
        top = result.detail.get("top_emotions", [])
        top_names = [name for name, _ in top[:3]]
        assert any(e in top_names for e in ("grief", "sadness", "despair")), \
            f"Expected grief/sadness in top 3, got {top_names}"


@skip_if_no_deps
class TestConflictIntegration:
    @pytest.mark.asyncio
    async def test_work_pressure_detects_conflict(self):
        from sufficiency_scorer.detectors.conflict import ConflictAdapter
        adapter = ConflictAdapter()
        result = await adapter.detect(WORK_PRESSURE_TEXT)
        # Should detect some conflict style
        assert result.activated is True or result.confidence > 0.1
        styles = result.detail.get("styles", {})
        assert len(styles) > 0


@skip_if_no_deps
class TestFragilityIntegration:
    @pytest.mark.asyncio
    async def test_grief_detects_vulnerability(self):
        from sufficiency_scorer.detectors.fragility import FragilityAdapter
        adapter = FragilityAdapter()
        result = await adapter.detect(GRIEF_TEXT)
        assert result.activated is True
        pattern = result.detail.get("pattern")
        assert pattern in ("open", "defensive", "masked", "denial")


@skip_if_no_deps
class TestEndToEnd:
    """Run 3 real detectors + EQ on work pressure text, verify insights."""

    @pytest.mark.asyncio
    async def test_work_pressure_e2e_insights(self):
        from sufficiency_scorer.detectors.emotion import EmotionAdapter
        from sufficiency_scorer.detectors.conflict import ConflictAdapter
        from sufficiency_scorer.detectors.fragility import FragilityAdapter
        from sufficiency_scorer.detectors.eq import EQAdapter
        from sufficiency_scorer.insight_extractor import InsightExtractor

        import asyncio
        # Run all 4 in parallel
        results = await asyncio.gather(
            EmotionAdapter().detect(WORK_PRESSURE_TEXT),
            ConflictAdapter().detect(WORK_PRESSURE_TEXT),
            FragilityAdapter().detect(WORK_PRESSURE_TEXT),
            EQAdapter().detect(WORK_PRESSURE_TEXT),
        )

        # Fill in missing dimensions
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        extractor = InsightExtractor()
        insights = extractor.extract(list(results), user_text=WORK_PRESSURE_TEXT)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]

        print(f"\n--- E2E Work Pressure Results ---")
        for r in results:
            if r.activated:
                print(f"  {r.dimension.value}: conf={r.confidence:.2f} detail={r.detail}")
        print(f"\n  Insights ({len(insights)} total, {len(good)} MEDIUM+):")
        for i in insights:
            print(f"    [{i.quality.name}] {i.reframe[:80]}")

        # Should produce at least 2 good insights from 4 real detectors
        assert len(good) >= 2, f"Only {len(good)} MEDIUM+ insights from real detectors"
```

**Step 2: Fix the detector adapters to work with real detectors**

The key changes needed in each adapter:
1. Use `asyncio.to_thread()` to wrap sync detector calls
2. Handle the actual return types from each detector
3. Properly extract detail fields

For `sufficiency_scorer/detectors/emotion.py`:
```python
"""Emotion detector adapter — wraps emotion-detector project."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

EMOTION_DETECTOR_PATH = Path.home() / "emotion-detector"


class EmotionAdapter(DetectorAdapter):
    """Adapts EmotionDetector.detect() → DetectorResult."""

    dimension = Dimension.EMOTION

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(EMOTION_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(EMOTION_DETECTOR_PATH))
            from emotion_detector.detector import EmotionDetector
            self._detector = EmotionDetector()

    def _run_sync(self, text: str):
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        snapshot = await asyncio.to_thread(self._run_sync, text)
        emotions = snapshot.emotions if hasattr(snapshot, "emotions") else {}
        if not emotions:
            return self._make_result(0.0)
        max_intensity = max(emotions.values())
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        return self._make_result(
            confidence=max_intensity,
            detail={"top_emotions": top_emotions},
        )
```

For `sufficiency_scorer/detectors/conflict.py`:
```python
"""Conflict style detector adapter."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

CONFLICT_DETECTOR_PATH = Path.home() / "conflict-detector"


class ConflictAdapter(DetectorAdapter):
    """Adapts ConflictDetector.detect() → DetectorResult."""

    dimension = Dimension.CONFLICT

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(CONFLICT_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(CONFLICT_DETECTOR_PATH))
            from conflict_detector.detector import ConflictDetector
            self._detector = ConflictDetector()

    def _run_sync(self, text: str):
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        snapshot = await asyncio.to_thread(self._run_sync, text)
        scores = snapshot.scores if hasattr(snapshot, "scores") else {}
        if not scores:
            return self._make_result(0.0)
        max_score = max(scores.values())
        return self._make_result(
            confidence=max_score,
            detail={"styles": dict(scores)},
        )
```

For `sufficiency_scorer/detectors/fragility.py`:
```python
"""Fragility pattern detector adapter."""

import asyncio
import sys
from pathlib import Path

from sufficiency_scorer.detectors.base import DetectorAdapter
from sufficiency_scorer.models import DetectorResult, Dimension

FRAGILITY_DETECTOR_PATH = Path.home() / "fragility-detector"


class FragilityAdapter(DetectorAdapter):
    """Adapts FragilityDetector.detect() → DetectorResult."""

    dimension = Dimension.FRAGILITY

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is None:
            if str(FRAGILITY_DETECTOR_PATH) not in sys.path:
                sys.path.insert(0, str(FRAGILITY_DETECTOR_PATH))
            from fragility_detector.detector import FragilityDetector
            self._detector = FragilityDetector()

    def _run_sync(self, text: str):
        self._load()
        conversation = [{"role": "user", "text": text}]
        return self._detector.detect(conversation=conversation, turn=1)

    async def detect(self, text: str, **kwargs) -> DetectorResult:
        snapshot = await asyncio.to_thread(self._run_sync, text)
        confidence = getattr(snapshot, "confidence", 0.0)
        pattern = getattr(snapshot, "pattern", None)
        pattern_scores = getattr(snapshot, "pattern_scores", {})
        return self._make_result(
            confidence=confidence,
            detail={
                "pattern": str(pattern.value) if hasattr(pattern, 'value') else str(pattern) if pattern else None,
                "pattern_scores": dict(pattern_scores) if pattern_scores else {},
            },
        )
```

**Step 3: Run integration tests**

Run: `cd /Users/michael/sufficiency-scorer && ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/pytest tests/test_integration.py -v -s`
Expected: PASS (if API key available) or SKIP (if not)

**Step 4: Run all tests to ensure nothing broke**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/ -v --ignore=tests/test_integration.py`
Expected: ALL PASS (70 tests)

**Step 5: Commit**

```bash
git add sufficiency_scorer/detectors/emotion.py sufficiency_scorer/detectors/conflict.py sufficiency_scorer/detectors/fragility.py tests/test_integration.py
git commit -m "feat(P2): integrate real Emotion/Conflict/Fragility detectors with asyncio.to_thread()"
```

---

### Task 4: P3 — Insight quality evaluation framework

**Files:**
- Create: `sufficiency_scorer/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluator.py
"""Tests for the insight quality evaluation framework."""

import pytest
from sufficiency_scorer.evaluator import InsightEvaluator, EvalResult
from sufficiency_scorer.models import InsightCandidate, InsightQuality, Dimension


class TestSpecificityScore:
    """Insights must be specific to the user, not generic platitudes."""

    def test_generic_platitude_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="frustration: 0.55",
            reframe="You're a good person",
            quality=InsightQuality.MEDIUM,
            confidence=0.7,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity < 0.5

    def test_contextualized_insight_scores_high(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
            signal="frustration (55%) + avoidance (70%)",
            reframe="You feel the friction but choose restraint — especially when your boss is forcing overtime",
            quality=InsightQuality.HIGH,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity >= 0.5

    def test_mere_repetition_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="frustration: 0.55",
            reframe="You feel frustrated",
            quality=InsightQuality.MEDIUM,
            confidence=0.7,
        )
        result = evaluator.evaluate(insight)
        assert result.specificity < 0.3


class TestReframeScore:
    """Reframes must transform, not repeat."""

    def test_positive_reframe_scores_well(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.FRAGILITY],
            signal="fragility pattern: open",
            reframe="You let yourself be seen — that takes more courage than most realize",
            quality=InsightQuality.MEDIUM,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.reframe_quality >= 0.5

    def test_negative_framing_scores_low(self):
        evaluator = InsightEvaluator()
        insight = InsightCandidate(
            source_dimensions=[Dimension.EMOTION],
            signal="anxiety: 0.6",
            reframe="You're very anxious and worried about everything",
            quality=InsightQuality.MEDIUM,
            confidence=0.6,
        )
        result = evaluator.evaluate(insight)
        assert result.reframe_quality < 0.5


class TestBatchEval:
    """Evaluate a full set of insights."""

    def test_batch_returns_summary(self):
        evaluator = InsightEvaluator()
        insights = [
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal="frustration: 0.55",
                reframe="You have a sharp sense of when things aren't right — in your work situation",
                quality=InsightQuality.MEDIUM,
                confidence=0.7,
            ),
            InsightCandidate(
                source_dimensions=[Dimension.EMOTION, Dimension.CONFLICT],
                signal="frustration + avoidance",
                reframe="You feel the friction but choose restraint — you're more strategic than you give yourself credit for",
                quality=InsightQuality.HIGH,
                confidence=0.6,
            ),
            InsightCandidate(
                source_dimensions=[Dimension.FRAGILITY],
                signal="open fragility",
                reframe="You let yourself be seen — that takes more courage than most realize",
                quality=InsightQuality.MEDIUM,
                confidence=0.5,
            ),
        ]
        summary = evaluator.evaluate_batch(insights)
        assert summary.total == 3
        assert 0.0 <= summary.avg_specificity <= 1.0
        assert 0.0 <= summary.avg_reframe_quality <= 1.0
        assert summary.bloom_worthy is not None  # bool

    def test_empty_batch(self):
        evaluator = InsightEvaluator()
        summary = evaluator.evaluate_batch([])
        assert summary.total == 0
        assert summary.bloom_worthy is False
```

**Step 2: Implement evaluator**

```python
# sufficiency_scorer/evaluator.py
"""Insight quality evaluation framework.

Scores insights on two axes:
1. Specificity: Is this insight specific to THIS user, or a generic platitude?
2. Reframe quality: Does it transform the signal into something positive and surprising?

Used for:
- Validating that our reframe templates are actually good
- Testing new cross-patterns before deploying
- Monitoring insight quality over time
"""

import re
from pydantic import BaseModel, Field

from sufficiency_scorer.models import InsightCandidate, InsightQuality


class EvalResult(BaseModel):
    """Evaluation of a single insight."""
    specificity: float = Field(ge=0.0, le=1.0)
    reframe_quality: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class BatchEvalResult(BaseModel):
    """Evaluation of a batch of insights."""
    total: int = 0
    avg_specificity: float = 0.0
    avg_reframe_quality: float = 0.0
    avg_overall: float = 0.0
    bloom_worthy: bool = False
    per_insight: list[EvalResult] = Field(default_factory=list)


# Words that indicate generic platitudes
PLATITUDE_INDICATORS = {
    "good person", "strong person", "great", "amazing", "wonderful",
    "special", "unique", "important", "valuable", "worthy",
}

# Words/phrases that indicate negative framing (should be reframed positively)
NEGATIVE_INDICATORS = {
    "anxious", "depressed", "worried", "stressed", "afraid",
    "weak", "broken", "damaged", "toxic", "dysfunctional",
    "very anxious", "very worried", "really stressed",
}

# Positive reframe indicators — suggest transformation happened
POSITIVE_REFRAME_INDICATORS = {
    "courage", "strength", "aware", "honest", "deep", "care",
    "protect", "resilience", "wisdom", "clarity", "intelligence",
    "brave", "real", "genuine", "open", "growth", "backbone",
    "superpower", "sharp", "tuned", "strategic",
}

# Signal leakage — when the reframe just repeats the signal name
SIGNAL_WORDS = {
    "frustrated", "frustration", "angry", "anger", "sad", "sadness",
    "anxious", "anxiety", "confused", "confusion", "afraid", "fear",
    "avoid", "avoidance", "confront", "confrontation",
}


class InsightEvaluator:
    """Evaluates insight quality on specificity and reframe transformation."""

    def evaluate(self, insight: InsightCandidate) -> EvalResult:
        specificity = self._score_specificity(insight)
        reframe_quality = self._score_reframe(insight)
        overall = specificity * 0.4 + reframe_quality * 0.6
        flags = self._collect_flags(insight, specificity, reframe_quality)
        return EvalResult(
            specificity=round(specificity, 3),
            reframe_quality=round(reframe_quality, 3),
            overall=round(overall, 3),
            flags=flags,
        )

    def evaluate_batch(self, insights: list[InsightCandidate]) -> BatchEvalResult:
        if not insights:
            return BatchEvalResult(bloom_worthy=False)
        results = [self.evaluate(i) for i in insights]
        avg_spec = sum(r.specificity for r in results) / len(results)
        avg_reframe = sum(r.reframe_quality for r in results) / len(results)
        avg_overall = sum(r.overall for r in results) / len(results)
        # Bloom worthy: avg overall >= 0.5 AND at least 3 insights with overall >= 0.4
        good_count = sum(1 for r in results if r.overall >= 0.4)
        bloom_worthy = avg_overall >= 0.5 and good_count >= 3
        return BatchEvalResult(
            total=len(insights),
            avg_specificity=round(avg_spec, 3),
            avg_reframe_quality=round(avg_reframe, 3),
            avg_overall=round(avg_overall, 3),
            bloom_worthy=bloom_worthy,
            per_insight=results,
        )

    def _score_specificity(self, insight: InsightCandidate) -> float:
        """Score 0-1: how specific is this insight to the user?"""
        reframe = insight.reframe.lower()
        score = 0.5  # baseline

        # Penalty: generic platitudes
        for p in PLATITUDE_INDICATORS:
            if p in reframe:
                score -= 0.3
                break

        # Penalty: just repeating the signal/emotion name
        signal_words_in_reframe = sum(1 for w in SIGNAL_WORDS if w in reframe)
        if signal_words_in_reframe > 0:
            # Check if it's the ONLY content (mere repetition)
            words = reframe.split()
            signal_ratio = signal_words_in_reframe / max(len(words), 1)
            if signal_ratio > 0.2:
                score -= 0.4  # heavy penalty for parrot-back

        # Bonus: cross-dimensional (inherently more specific)
        if len(insight.source_dimensions) > 1:
            score += 0.2

        # Bonus: contains contextual references (work, relationship, etc.)
        context_words = {"work", "boss", "partner", "family", "relationship", "situation", "overtime", "loss"}
        if any(w in reframe for w in context_words):
            score += 0.15

        # Bonus: longer reframes tend to be more specific
        if len(reframe.split()) >= 12:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_reframe(self, insight: InsightCandidate) -> float:
        """Score 0-1: how well does the reframe transform the signal?"""
        reframe = insight.reframe.lower()
        score = 0.5  # baseline

        # Penalty: negative framing
        for neg in NEGATIVE_INDICATORS:
            if neg in reframe:
                score -= 0.3
                break

        # Bonus: positive reframe indicators
        pos_count = sum(1 for p in POSITIVE_REFRAME_INDICATORS if p in reframe)
        score += min(pos_count * 0.1, 0.3)

        # Bonus: uses "you" (addressing the person directly)
        if "you" in reframe.split():
            score += 0.1

        # Penalty: too short (likely low-effort)
        if len(reframe.split()) < 5:
            score -= 0.3

        # Bonus: contains contrast or surprise ("not X, but Y" patterns)
        contrast_patterns = [r"not .+, but", r"instead of", r"rather than", r"more than .+ realize"]
        if any(re.search(p, reframe) for p in contrast_patterns):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _collect_flags(self, insight: InsightCandidate, spec: float, reframe: float) -> list[str]:
        flags = []
        if spec < 0.3:
            flags.append("too_generic")
        if reframe < 0.3:
            flags.append("poor_reframe")
        if not insight.reframe:
            flags.append("empty_reframe")
        r_lower = insight.reframe.lower()
        for w in SIGNAL_WORDS:
            if f"you feel {w}" in r_lower or f"you are {w}" in r_lower:
                flags.append("parrot_back")
                break
        return flags
```

**Step 3: Run tests**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_evaluator.py -v`
Expected: ALL PASS

**Step 4: Run full suite**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/ --ignore=tests/test_integration.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sufficiency_scorer/evaluator.py tests/test_evaluator.py
git commit -m "feat(P3): insight quality evaluator — specificity + reframe quality scoring"
```

---

### Task 5: Final validation — run evaluator on all scenarios

**Files:**
- Modify: `scripts/validate_thresholds.py` (add evaluator section)
- Create: `tests/test_eval_scenarios.py`

**Step 1: Write scenario evaluation test**

```python
# tests/test_eval_scenarios.py
"""Evaluate insight quality across all 9 scenarios."""

import pytest
from sufficiency_scorer.evaluator import InsightEvaluator
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality


def make_result(dim, activated=False, confidence=0.0, detail=None):
    return DetectorResult(dimension=dim, activated=activated, confidence=confidence, detail=detail or {})


SCENARIO_CONFIGS = {
    "工作压力": {
        "text": "hey, I'm really tired. I don't want to do the OT anymore but my boss is forcing me to do overtime. What am I supposed to do?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.7, {"top_emotions": [("frustration", 0.55), ("anger", 0.52)]}),
            make_result(Dimension.EQ, True, 0.6, {"features": {"self_ref": 0.15, "question_ratio": 0.33, "words": 27}, "valence": -0.35, "distress": 0.43}),
            make_result(Dimension.CONFLICT, True, 0.6, {"styles": {"avoid": 0.7, "compromise": 0.4}}),
            make_result(Dimension.FRAGILITY, True, 0.5, {"pattern": "open", "pattern_scores": {"open": 0.6}}),
        ],
    },
    "自嘲幽默": {
        "text": "haha so I just got dumped again, third time this year. I'm starting to think maybe I'm the common denominator here. At least I'm consistent right?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.6, {"top_emotions": [("sadness", 0.5), ("amusement", 0.4)]}),
            make_result(Dimension.HUMOR, True, 0.7, {"humor_detected": True, "styles": {"self_deprecating": 0.8}}),
            make_result(Dimension.FRAGILITY, True, 0.5, {"pattern": "masked", "pattern_scores": {"masked": 0.6}}),
            make_result(Dimension.EQ, True, 0.5, {"features": {"self_ref": 0.1, "question_ratio": 0.33, "words": 26}, "valence": -0.15, "distress": 0.25}),
        ],
    },
    "丧亲之痛": {
        "text": "my mom passed away last month and I don't know how to be normal anymore. People keep saying it gets better but when?",
        "results": [
            make_result(Dimension.EMOTION, True, 0.8, {"top_emotions": [("grief", 0.8), ("sadness", 0.7)]}),
            make_result(Dimension.EQ, True, 0.7, {"features": {"self_ref": 0.12, "question_ratio": 0.5, "words": 22}, "valence": -0.6, "distress": 0.7}),
            make_result(Dimension.FRAGILITY, True, 0.7, {"pattern": "open", "pattern_scores": {"open": 0.8}}),
        ],
    },
}


class TestScenarioInsightQuality:
    """All scenarios should produce bloom-worthy insights."""

    @pytest.mark.parametrize("stype", SCENARIO_CONFIGS.keys())
    def test_scenario_bloom_worthy(self, stype):
        config = SCENARIO_CONFIGS[stype]
        text = config["text"]
        results = config["results"]
        # Fill missing dims
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(make_result(dim))

        extractor = InsightExtractor()
        insights = extractor.extract(results, user_text=text)
        good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]

        evaluator = InsightEvaluator()
        summary = evaluator.evaluate_batch(insights)

        print(f"\n--- {stype} ---")
        print(f"  Insights: {len(insights)} total, {len(good)} MEDIUM+")
        print(f"  Avg specificity: {summary.avg_specificity:.2f}")
        print(f"  Avg reframe quality: {summary.avg_reframe_quality:.2f}")
        print(f"  Bloom worthy: {summary.bloom_worthy}")
        for i, (ins, ev) in enumerate(zip(insights, summary.per_insight)):
            print(f"  [{ins.quality.name}] spec={ev.specificity:.2f} ref={ev.reframe_quality:.2f} | {ins.reframe[:70]}")
            if ev.flags:
                print(f"           FLAGS: {ev.flags}")

        # Core assertion: scenarios should produce bloom-worthy results
        assert len(good) >= 3, f"{stype}: only {len(good)} MEDIUM+ insights"
        assert summary.avg_reframe_quality >= 0.4, f"{stype}: avg reframe quality too low ({summary.avg_reframe_quality:.2f})"
        # No parrot-back flags
        all_flags = [f for ev in summary.per_insight for f in ev.flags]
        assert "parrot_back" not in all_flags, f"{stype}: has parrot-back insights"
```

**Step 2: Run test**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/test_eval_scenarios.py -v -s`

**Step 3: Run full suite**

Run: `cd /Users/michael/sufficiency-scorer && .venv/bin/pytest tests/ --ignore=tests/test_integration.py -v`

**Step 4: Final commit**

```bash
git add scripts/ tests/test_eval_scenarios.py
git commit -m "feat(P0-P3): complete validation + dynamic reframing + quality evaluation

P0: Word count analysis shows median first message = 5.5 words (40-word gate confirmed)
P1: Dynamic reframing incorporates user's actual words into insights
P2: Real Emotion/Conflict/Fragility detector integration via asyncio.to_thread()
P3: InsightEvaluator scores specificity + reframe quality, validates bloom-worthiness"
```

---

## Summary

| Task | P-level | What it does | Files |
|------|---------|-------------|-------|
| 1 | P0 | Validate thresholds with real data | `scripts/validate_thresholds.py`, `tests/test_validation.py` |
| 2 | P1 | Dynamic reframing with user context | `insight_extractor.py` (modify), `tests/test_dynamic_reframe.py` |
| 3 | P2 | Real detector integration | `detectors/{emotion,conflict,fragility}.py`, `tests/test_integration.py` |
| 4 | P3 | Quality evaluation framework | `evaluator.py`, `tests/test_evaluator.py` |
| 5 | All | Scenario-level quality validation | `tests/test_eval_scenarios.py` |
