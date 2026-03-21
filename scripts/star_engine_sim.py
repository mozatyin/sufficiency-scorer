"""Simulate 100 users through StarEngine over 4 conversation turns.

Each literary character = one user. Their first-meeting text = turn 1.
Turns 2-4 are synthesized continuations based on their character arc.

Metrics:
  - Stars per user at each turn
  - Label accuracy (does the label match character's ground truth?)
  - Minimum guarantee hit rate
  - Dark star trigger rate
  - Label diversity (are we giving different labels or repeating?)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.literary_validation import (
    heuristic_emotion, heuristic_conflict, heuristic_humor, heuristic_fragility,
)
# Also import the expanded keyword sets that were improved during PDCA R3

from sufficiency_scorer.detectors.eq import EQAdapter, extract_behavioral
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.star_engine import StarEngine
from sufficiency_scorer.star_labels import get_dark_labels
import asyncio

DATA = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"
GT = Path(__file__).resolve().parent.parent / "data" / "literary_ground_truth.json"


def simulate_later_turns(char: dict, gt: dict) -> list[str]:
    """Generate personalized turn 2-4 text from character's deep truths.

    Each turn reveals a different aspect of the character, ensuring
    different dimensions activate across turns (not the same ones repeating).
    """
    deep = gt.get("deep_truths", [])

    # Turn 2: Use the character's FIRST deep truth as basis for emotional disclosure
    if len(deep) >= 1:
        t2 = f"You know what, {deep[0]}. I've never said that out loud before. It's been eating at me."
    else:
        t2 = "There's something I haven't told you. Something I've been carrying for a long time."

    # Turn 3: Use the character's SECOND deep truth — conflict/action oriented
    if len(deep) >= 2:
        t3 = f"And the thing is, {deep[1]}. I keep going back and forth about what to do about it. Part of me wants to fight, part of me wants to run."
    else:
        t3 = "I've been going back and forth about what to do. Part of me wants to fight, part of me wants to run."

    # Turn 4: Use the character's THIRD deep truth — vulnerability + resolution
    if len(deep) >= 3:
        t4 = f"I think deep down, {deep[2]}. That's what scares me and gives me hope at the same time. Does that make sense?"
    else:
        t4 = "I think I'm starting to see what I really need. It scares me but it also feels right. Does that make sense?"

    return [t2, t3, t4]


def build_results(text: str, eq_result: DetectorResult) -> list[DetectorResult]:
    """Build detector results from heuristics + real EQ."""
    results = [
        heuristic_emotion(text),
        eq_result,
        heuristic_conflict(text),
        heuristic_humor(text),
        heuristic_fragility(text),
    ]
    seen = {r.dimension for r in results}
    for dim in Dimension:
        if dim not in seen:
            results.append(DetectorResult(dimension=dim))
    return results


def label_matches_gt(label: str, gt: dict) -> str:
    """Check if star label aligns with character ground truth.
    Returns: 'deep', 'surface', 'wrong', 'generic'
    """
    label_lower = label.lower()

    # Check deep truths
    for deep in gt.get("deep_truths", []):
        deep_lower = deep.lower()
        # Extract key concepts
        concepts = [w for w in deep_lower.split() if len(w) > 3 and w not in
                    {"that", "this", "with", "from", "have", "been", "their", "about", "which", "when", "what", "more", "than", "into", "just", "very", "also", "some"}]
        hits = sum(1 for c in concepts if c in label_lower)
        if hits >= 1:
            return "deep"

    # Check surface truths
    for surface in gt.get("surface_truths", []):
        surface_lower = surface.lower()
        concepts = [w for w in surface_lower.split() if len(w) > 3]
        hits = sum(1 for c in concepts if c in label_lower)
        if hits >= 1:
            return "surface"

    # Check wrong claims
    for wrong in gt.get("wrong_claims", []):
        wrong_lower = wrong.lower()
        concepts = [w for w in wrong_lower.split() if len(w) > 3]
        hits = sum(1 for c in concepts if c in label_lower)
        if hits >= 1:
            return "wrong"

    return "generic"


async def main():
    with open(DATA) as f:
        characters = json.load(f)
    with open(GT) as f:
        ground_truths = {g["id"]: g for g in json.load(f)}

    eq = EQAdapter()

    # === Per-user metrics ===
    user_results = []
    all_labels = []
    stars_by_turn = {1: [], 2: [], 3: [], 4: []}
    label_quality = {"deep": 0, "surface": 0, "generic": 0, "wrong": 0}
    min_guarantee_failures = {2: 0, 4: 0}
    dark_star_count = 0
    fog_event_count = 0
    total_users = 0

    for char in characters:
        gt = ground_truths.get(char["id"])
        if not gt:
            continue
        total_users += 1

        engine = StarEngine()
        turns = [char["text"]] + simulate_later_turns(char, gt)

        user_data = {
            "id": char["id"],
            "character": char["character"],
            "stars_per_turn": [],
            "labels": [],
            "label_quality": [],
            "dark_stars": 0,
            "fog_events": 0,
        }

        for turn_idx, text in enumerate(turns):
            turn = turn_idx + 1
            eq_result = await eq.detect(text)
            results = build_results(text, eq_result)

            output = engine.process_turn(results, turn_count=turn, user_text=text)

            user_data["stars_per_turn"].append(len(engine.stars))
            user_data["fog_events"] += len(output.fog_events)
            fog_event_count += len(output.fog_events)

            for star_event in output.new_stars:
                star = star_event.star
                lbl = star.label
                user_data["labels"].append(lbl)
                all_labels.append(lbl)

                if star.is_dark:
                    user_data["dark_stars"] += 1
                    dark_star_count += 1
                else:
                    quality = label_matches_gt(lbl, gt)
                    user_data["label_quality"].append(quality)
                    label_quality[quality] += 1

            stars_by_turn[turn].append(len(engine.stars))

        # Check minimum guarantees
        if len(engine.stars) < 2 and user_data["stars_per_turn"][1] < 2:
            min_guarantee_failures[2] += 1
        if len(engine.stars) < 3 and len(user_data["stars_per_turn"]) >= 4 and user_data["stars_per_turn"][3] < 3:
            min_guarantee_failures[4] += 1

        user_results.append(user_data)

        icon = "★" if len(engine.stars) >= 4 else "✓" if len(engine.stars) >= 3 else "○" if len(engine.stars) >= 2 else "·"
        stars_str = "→".join(str(s) for s in user_data["stars_per_turn"])
        print(f"  {icon} #{char['id']:>3} {char['character']:<25} stars={stars_str} labels={user_data['labels'][:3]}")

    # === REPORT ===
    print("\n" + "=" * 80)
    print("STAR ENGINE SIMULATION — 100 USERS × 4 TURNS")
    print("=" * 80)

    print(f"\n--- Stars Per Turn (avg across {total_users} users) ---")
    for turn in [1, 2, 3, 4]:
        vals = stars_by_turn[turn]
        avg = sum(vals) / len(vals) if vals else 0
        has_min = {1: "-", 2: "≥2", 3: "-", 4: "≥3"}
        pct_met = sum(1 for v in vals if v >= {1:0, 2:2, 3:0, 4:3}.get(turn, 0)) / len(vals) * 100 if vals else 0
        bar = "█" * int(avg * 5)
        print(f"  Turn {turn}: avg={avg:.1f} {bar}  (min={has_min[turn]}, met={pct_met:.0f}%)")

    print(f"\n--- Label Quality ---")
    total_labels = sum(label_quality.values())
    for q in ["deep", "surface", "generic", "wrong"]:
        n = label_quality[q]
        pct = n / total_labels * 100 if total_labels else 0
        print(f"  {q:>8}: {n:>4} ({pct:.0f}%)")

    print(f"\n--- Label Diversity ---")
    label_counts = Counter(all_labels)
    unique = len(label_counts)
    total = len(all_labels)
    print(f"  Total labels used: {total}")
    print(f"  Unique labels: {unique}")
    print(f"  Diversity ratio: {unique/total*100:.0f}%")
    print(f"  Most common:")
    for lbl, count in label_counts.most_common(10):
        print(f"    {count:>3}x  {lbl}")

    print(f"\n--- Special Events ---")
    print(f"  Dark stars: {dark_star_count} ({dark_star_count/total_users*100:.0f}% of users)")
    print(f"  Fog events: {fog_event_count} (avg {fog_event_count/total_users:.1f}/user)")
    print(f"  Min guarantee failures: turn2={min_guarantee_failures[2]}, turn4={min_guarantee_failures[4]}")

    # Bottom 10 (fewest stars)
    user_results.sort(key=lambda u: u["stars_per_turn"][-1])
    print(f"\n--- Bottom 10 (Fewest Stars After 4 Turns) ---")
    for u in user_results[:10]:
        stars_str = "→".join(str(s) for s in u["stars_per_turn"])
        print(f"  #{u['id']:>3} {u['character']:<25} {stars_str} labels={u['labels']}")

    # Top 10 (most stars)
    print(f"\n--- Top 10 (Most Stars After 4 Turns) ---")
    for u in user_results[-10:]:
        stars_str = "→".join(str(s) for s in u["stars_per_turn"])
        print(f"  #{u['id']:>3} {u['character']:<25} {stars_str} labels={u['labels'][:4]}")

    # Save
    output_path = Path(__file__).resolve().parent.parent / "data" / "star_engine_sim.json"
    with open(output_path, "w") as f:
        json.dump(user_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
