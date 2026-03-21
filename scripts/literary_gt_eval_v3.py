"""GT eval using V3 minimal generator — precomputed signals + short prompt."""
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.literary_gt_eval_llm_judge import LLMJudge
from sufficiency_scorer.insight_generator import InsightGenerator as InsightGeneratorV3

DATA = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"
GT = Path(__file__).resolve().parent.parent / "data" / "literary_ground_truth.json"

async def main():
    with open(DATA) as f: chars = json.load(f)
    with open(GT) as f: gts = {g["id"]: g for g in json.load(f)}

    gen = InsightGeneratorV3()
    judge = LLMJudge()
    all_scores, gen_times = [], []

    for c in chars:
        gt = gts.get(c["id"])
        if not gt: continue
        t0 = time.monotonic()
        try:
            insights = gen.generate(c["text"])
        except Exception as e:
            print(f"  ERR #{c['id']}: {e}"); continue
        gen_times.append(time.monotonic() - t0)

        scores = []
        for ins in insights[:3]:
            try:
                r = judge.judge(c["character"], c["novel"], c["text"], gt["deep_truths"], ins.reframe)
                scores.append(r["score"]); all_scores.append(r["score"])
            except: pass

        avg = sum(scores)/len(scores) if scores else 0
        icon = "★" if avg >= 85 else "✓" if avg >= 70 else "○" if avg >= 50 else "·"
        print(f"  {icon} #{c['id']:>3} {c['character']:<25} avg={avg:.0f} gen={gen_times[-1]*1000:.0f}ms")

    n = len(all_scores)
    deep = sum(1 for s in all_scores if s >= 80)
    wrong = sum(1 for s in all_scores if s < 20)
    print(f"\n{'='*60}")
    print(f"V3 SONNET MINIMAL: Mean={sum(all_scores)/n:.1f} Median={sorted(all_scores)[n//2]}")
    print(f"DEEP={deep}/{n} ({deep/n*100:.0f}%) WRONG={wrong}/{n} ({wrong/n*100:.0f}%)")
    print(f"Gen: avg={sum(gen_times)/len(gen_times)*1000:.0f}ms min={min(gen_times)*1000:.0f}ms")

if __name__ == "__main__":
    asyncio.run(main())
