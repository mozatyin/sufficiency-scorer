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
    sessions_3: dict[str, list[str]] = {}
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
