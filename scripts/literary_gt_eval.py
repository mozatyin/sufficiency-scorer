"""Evaluate insights against literary character ground truth.

Each character scores our insights:
- 80-100: Matches deep_truths (correct + surprising = "how did you know?!")
- 40-70:  Matches surface_truths (correct but obvious, they already knew)
- 10-30:  Doesn't match anything (generic, neither wrong nor right)
- 0-10:   Matches wrong_claims (incorrect, would feel misunderstood)
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.literary_validation import (
    analyze_character,
)
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.evaluator import InsightEvaluator

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"
GT_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_ground_truth.json"


def score_insight_against_gt(reframe: str, gt: dict) -> tuple[int, str]:
    """Score a single reframe against character's ground truth.

    Returns (score 0-100, reason).
    """
    reframe_lower = reframe.lower()

    # Check wrong_claims first (worst case)
    for wrong in gt.get("wrong_claims", []):
        if _matches(reframe_lower, wrong):
            return 5, f"WRONG: matches '{wrong}'"

    # Check deep_truths (best case — "how did you know?!")
    best_deep_score = 0
    best_deep_match = ""
    for deep in gt.get("deep_truths", []):
        match_strength = _match_strength(reframe_lower, deep)
        if match_strength > best_deep_score:
            best_deep_score = match_strength
            best_deep_match = deep

    if best_deep_score >= 0.4:
        score = int(80 + best_deep_score * 20)  # 80-100
        return min(score, 100), f"DEEP: '{best_deep_match}'"

    # Check surface_truths (correct but obvious)
    for surface in gt.get("surface_truths", []):
        if _matches(reframe_lower, surface):
            return 50, f"SURFACE: '{surface}'"

    # Partial deep match
    if best_deep_score >= 0.2:
        score = int(60 + best_deep_score * 50)  # 60-70
        return min(score, 75), f"PARTIAL DEEP: '{best_deep_match}'"

    # Generic — not wrong but not insightful
    return 25, "GENERIC: no ground truth match"


def _matches(reframe: str, truth: str) -> bool:
    """Check if a reframe aligns with a truth statement."""
    truth_lower = truth.lower()
    concepts = _extract_concepts(truth_lower)
    if not concepts:
        return False
    hits = sum(1 for c in concepts if c in reframe)
    # Need at least 2 concept matches AND >20% concept coverage
    # This prevents single-word false positives like "self-aware" matching "not self-aware"
    ratio = hits / len(concepts) if concepts else 0
    return hits >= 2 and ratio >= 0.15


def _match_strength(reframe: str, truth: str) -> float:
    """How strongly does a reframe match a deep truth? 0.0-1.0"""
    truth_lower = truth.lower()
    concepts = _extract_concepts(truth_lower)
    if not concepts:
        return 0.0
    hits = sum(1 for c in concepts if c in reframe)
    return hits / len(concepts)


def _extract_concepts(text: str) -> list[str]:
    """Extract key conceptual words/phrases from a truth statement."""
    # Remove common stop words
    stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "about",
            "not", "but", "and", "or", "nor", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more", "most",
            "other", "some", "such", "no", "only", "own", "same", "than",
            "too", "very", "just", "because", "that", "this", "these", "those",
            "it", "its", "he", "she", "his", "her", "they", "them", "their",
            "who", "whom", "which", "what", "when", "where", "how", "why",
            "if", "then", "else", "while", "until", "although", "though"}

    words = re.findall(r'\b[a-z]+\b', text)
    concepts = [w for w in words if w not in stop and len(w) > 3]

    # Also extract 2-word phrases
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if words[i] not in stop or words[i+1] not in stop:
            if len(bigram) > 6:
                concepts.append(bigram)

    return concepts


async def main():
    with open(DATA_PATH) as f:
        characters = json.load(f)
    with open(GT_PATH) as f:
        ground_truths = {gt["id"]: gt for gt in json.load(f)}

    eq = EQAdapter()
    extractor = InsightExtractor()
    evaluator = InsightEvaluator()

    all_scores = []
    char_results = []

    for char in characters:
        cid = char["id"]
        gt = ground_truths.get(cid)
        if not gt:
            continue

        r = await analyze_character(char, eq, extractor, evaluator)

        insight_scores = []
        for ins in r["insights"]:
            score, reason = score_insight_against_gt(ins["reframe"], gt)
            insight_scores.append({
                "score": score,
                "reason": reason,
                "reframe": ins["reframe"][:80],
                "quality": ins["quality"],
            })
            all_scores.append(score)

        avg_score = sum(s["score"] for s in insight_scores) / len(insight_scores) if insight_scores else 0
        char_results.append({
            "id": cid,
            "character": char["character"],
            "novel": char["novel"][:30],
            "avg_score": avg_score,
            "insight_count": len(insight_scores),
            "scores": insight_scores,
        })

    # === Report ===
    print("=" * 90)
    print("GROUND TRUTH EVALUATION: How Would Each Character Score Our Insights?")
    print("=" * 90)

    char_results.sort(key=lambda x: x["avg_score"], reverse=True)

    # Overall
    print(f"\n--- Overall ---")
    print(f"  Characters evaluated: {len(char_results)}")
    print(f"  Total insights scored: {len(all_scores)}")
    if all_scores:
        print(f"  Mean score: {sum(all_scores)/len(all_scores):.1f}/100")
        print(f"  Median: {sorted(all_scores)[len(all_scores)//2]}/100")

    # Distribution
    deep = sum(1 for s in all_scores if s >= 80)
    partial = sum(1 for s in all_scores if 60 <= s < 80)
    surface = sum(1 for s in all_scores if 40 <= s < 60)
    generic = sum(1 for s in all_scores if 10 < s < 40)
    wrong = sum(1 for s in all_scores if s <= 10)
    print(f"\n--- Score Distribution ---")
    print(f"  80-100 (DEEP — 'how did you know?!'):  {deep:>3} ({deep/len(all_scores)*100:.0f}%)")
    print(f"  60-79  (PARTIAL DEEP):                  {partial:>3} ({partial/len(all_scores)*100:.0f}%)")
    print(f"  40-59  (SURFACE — correct but obvious): {surface:>3} ({surface/len(all_scores)*100:.0f}%)")
    print(f"  11-39  (GENERIC — meh):                 {generic:>3} ({generic/len(all_scores)*100:.0f}%)")
    print(f"   0-10  (WRONG — misunderstood):         {wrong:>3} ({wrong/len(all_scores)*100:.0f}%)")

    # Top 15 characters (most impressed)
    print(f"\n--- Top 15 Characters (Would Be Most Impressed) ---")
    for r in char_results[:15]:
        print(f"  {r['avg_score']:>5.0f}  #{r['id']:>3} {r['character']:<25} ({r['insight_count']} insights)")
        for s in r["scores"][:2]:
            print(f"         [{s['score']:>3}] {s['reason'][:50]}")
            print(f"               → {s['reframe'][:70]}")

    # Bottom 15 (least impressed)
    print(f"\n--- Bottom 15 Characters (Least Impressed) ---")
    for r in char_results[-15:]:
        print(f"  {r['avg_score']:>5.0f}  #{r['id']:>3} {r['character']:<25} ({r['insight_count']} insights)")
        for s in r["scores"][:2]:
            print(f"         [{s['score']:>3}] {s['reason'][:50]}")
            print(f"               → {s['reframe'][:70]}")

    # WRONG insights — the most critical failures
    wrongs = []
    for r in char_results:
        for s in r["scores"]:
            if s["score"] <= 10:
                wrongs.append({"character": r["character"], **s})

    if wrongs:
        print(f"\n--- WRONG Insights ({len(wrongs)}) — Characters Would Feel Misunderstood ---")
        for w in wrongs[:20]:
            print(f"  {w['character']:<25} [{w['score']}] {w['reason']}")
            print(f"    → {w['reframe'][:80]}")

    # Save
    output = Path(__file__).resolve().parent.parent / "data" / "literary_gt_scores.json"
    with open(output, "w") as f:
        json.dump(char_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
