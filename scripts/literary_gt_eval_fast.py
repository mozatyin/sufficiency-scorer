"""GT eval using fast Haiku generator — same judge, faster generation."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.literary_validation import (
    heuristic_emotion, heuristic_conflict, heuristic_humor, heuristic_fragility,
)
from scripts.literary_gt_eval_llm_judge import LLMJudge
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.insight_generator import InsightGenerator as FastInsightGenerator
from sufficiency_scorer.models import DetectorResult, Dimension

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"
GT_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_ground_truth.json"


async def main():
    with open(DATA_PATH) as f:
        characters = json.load(f)
    with open(GT_PATH) as f:
        ground_truths = {gt["id"]: gt for gt in json.load(f)}

    eq = EQAdapter()
    generator = FastInsightGenerator()
    judge = LLMJudge()

    all_scores = []
    gen_times = []
    char_results = []

    for char in characters:
        cid = char["id"]
        gt = ground_truths.get(cid)
        if not gt:
            continue

        text = char["text"]

        # Detectors
        eq_result = await eq.detect(text)
        results = [heuristic_emotion(text), eq_result, heuristic_conflict(text),
                   heuristic_humor(text), heuristic_fragility(text)]
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        # Generate (timed)
        t0 = time.monotonic()
        try:
            insights = generator.generate(user_text=text)
        except Exception as e:
            print(f"  GEN ERROR #{cid}: {e}")
            continue
        gen_time = time.monotonic() - t0
        gen_times.append(gen_time)

        # Judge
        insight_scores = []
        for ins in insights[:4]:
            try:
                result = judge.judge(
                    character=char["character"], novel=char["novel"],
                    text=text, deep_truths=gt["deep_truths"], reframe=ins.reframe,
                )
                insight_scores.append(result["score"])
                all_scores.append(result["score"])
            except Exception:
                pass

        avg = sum(insight_scores) / len(insight_scores) if insight_scores else 0
        char_results.append({"id": cid, "character": char["character"], "avg_score": avg})

        icon = "★" if avg >= 85 else "✓" if avg >= 70 else "○" if avg >= 50 else "·"
        print(f"  {icon} #{cid:>3} {char['character']:<25} avg={avg:.0f} gen={gen_time*1000:.0f}ms")

    # Report
    print("\n" + "=" * 80)
    print("FAST (HAIKU) vs STANDARD (SONNET) COMPARISON")
    print("=" * 80)

    if all_scores:
        n = len(all_scores)
        print(f"\n  Mean score:    {sum(all_scores)/n:.1f}/100")
        print(f"  Median:        {sorted(all_scores)[n//2]}/100")
        print(f"  Avg gen time:  {sum(gen_times)/len(gen_times)*1000:.0f}ms")
        print(f"  Min gen time:  {min(gen_times)*1000:.0f}ms")
        print(f"  Max gen time:  {max(gen_times)*1000:.0f}ms")

        deep = sum(1 for s in all_scores if s >= 80)
        wrong = sum(1 for s in all_scores if s < 20)
        print(f"\n  DEEP (80+):  {deep}/{n} ({deep/n*100:.0f}%)")
        print(f"  WRONG (<20): {wrong}/{n} ({wrong/n*100:.0f}%)")

    # Bottom 10
    char_results.sort(key=lambda x: x["avg_score"])
    print(f"\n  Bottom 10:")
    for r in char_results[:10]:
        print(f"    {r['avg_score']:>5.0f} #{r['id']:>3} {r['character']}")

    output = Path(__file__).resolve().parent.parent / "data" / "literary_gt_fast.json"
    with open(output, "w") as f:
        json.dump(char_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
