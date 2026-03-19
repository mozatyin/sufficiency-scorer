"""Ground truth eval using LLM-as-judge scoring.

Instead of keyword matching, use LLM to judge:
"If you are [character], and someone said [reframe] to you, how would you score it?"
"""

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
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.insight_generator import InsightGenerator
from sufficiency_scorer.models import DetectorResult, Dimension

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"
GT_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_ground_truth.json"

JUDGE_SYSTEM = """You are a literary character evaluating insights about yourself.

You will be given:
1. Your name and novel
2. What you said (your actual words)
3. Your TRUE inner reality (deep truths only you know)
4. An insight someone generated about you

Score the insight 0-100:
- 90-100: Perfectly captures a deep truth about me that I barely admit to myself. I'd be stunned. "How did you know?"
- 70-89: Touches on something genuinely true and non-obvious. I'd be impressed.
- 50-69: Correct observation but somewhat obvious from what I said. Not surprising.
- 30-49: Vaguely relevant but could apply to many people. Generic.
- 10-29: Misses the point. Not wrong but not insightful.
- 0-9: Actually wrong about me. I'd feel misunderstood.

Respond with ONLY a JSON object: {"score": N, "reason": "brief explanation"}"""

JUDGE_USER_TEMPLATE = """CHARACTER: {character} from "{novel}"

WHAT I SAID:
"{text}"

MY DEEP TRUTHS (what's really going on inside me):
{deep_truths}

INSIGHT TO EVALUATE:
"{reframe}"

Score this insight 0-100. JSON only."""


class LLMJudge:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        import anthropic
        if api_key.startswith("sk-or-"):
            self._client = anthropic.Anthropic(api_key=api_key, base_url="https://openrouter.ai/api")
            self._model = "anthropic/claude-sonnet-4"
        else:
            self._client = anthropic.Anthropic(api_key=api_key)
            self._model = "claude-sonnet-4-20250514"

    def judge(self, character: str, novel: str, text: str, deep_truths: list[str], reframe: str) -> dict:
        prompt = JUDGE_USER_TEMPLATE.format(
            character=character,
            novel=novel,
            text=text[:200],
            deep_truths="\n".join(f"- {t}" for t in deep_truths),
            reframe=reframe,
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text_resp = resp.content[0].text.strip()
        try:
            # Handle markdown code blocks
            if text_resp.startswith("```"):
                text_resp = text_resp.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text_resp)
            return {"score": int(result.get("score", 25)), "reason": result.get("reason", "")}
        except (json.JSONDecodeError, ValueError):
            import re
            match = re.search(r'"score"\s*:\s*(\d+)', text_resp)
            if match:
                return {"score": int(match.group(1)), "reason": text_resp[:100]}
            return {"score": 25, "reason": f"parse_error: {text_resp[:80]}"}


async def main():
    with open(DATA_PATH) as f:
        characters = json.load(f)
    with open(GT_PATH) as f:
        ground_truths = {gt["id"]: gt for gt in json.load(f)}

    eq = EQAdapter()
    generator = InsightGenerator()
    judge = LLMJudge()

    all_scores = []
    char_results = []

    for char in characters:
        cid = char["id"]
        gt = ground_truths.get(cid)
        if not gt:
            continue

        text = char["text"]

        # Run detectors
        eq_result = await eq.detect(text)
        results = [heuristic_emotion(text), eq_result, heuristic_conflict(text),
                   heuristic_humor(text), heuristic_fragility(text)]
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        # Generate insights
        try:
            insights = generator.generate(results, user_text=text)
        except Exception as e:
            print(f"  GEN ERROR #{cid}: {e}")
            continue

        # Judge each insight
        insight_scores = []
        for ins in insights[:4]:  # cap at 4 to save API calls
            try:
                result = judge.judge(
                    character=char["character"],
                    novel=char["novel"],
                    text=text,
                    deep_truths=gt["deep_truths"],
                    reframe=ins.reframe,
                )
                insight_scores.append({
                    "score": result["score"],
                    "reason": result["reason"][:80],
                    "reframe": ins.reframe[:100],
                })
                all_scores.append(result["score"])
            except Exception as e:
                print(f"  JUDGE ERROR #{cid}: {e}")

        avg = sum(s["score"] for s in insight_scores) / len(insight_scores) if insight_scores else 0
        char_results.append({
            "id": cid, "character": char["character"], "novel": char["novel"][:30],
            "avg_score": avg, "insight_count": len(insight_scores), "scores": insight_scores,
        })

        icon = "★" if avg >= 70 else "✓" if avg >= 50 else "○" if avg >= 30 else "·"
        print(f"  {icon} #{cid:>3} {char['character']:<25} avg={avg:.0f}")

    # Report
    print("\n" + "=" * 90)
    print("GROUND TRUTH EVAL — LLM JUDGE")
    print("=" * 90)

    char_results.sort(key=lambda x: x["avg_score"], reverse=True)

    if all_scores:
        print(f"\n--- Overall ---")
        print(f"  Mean: {sum(all_scores)/len(all_scores):.1f}/100")
        print(f"  Median: {sorted(all_scores)[len(all_scores)//2]}/100")
        print(f"  Total insights judged: {len(all_scores)}")

        deep = sum(1 for s in all_scores if s >= 80)
        partial = sum(1 for s in all_scores if 60 <= s < 80)
        surface = sum(1 for s in all_scores if 40 <= s < 60)
        generic = sum(1 for s in all_scores if 20 <= s < 40)
        wrong = sum(1 for s in all_scores if s < 20)
        n = len(all_scores)
        print(f"\n--- Distribution ---")
        print(f"  80-100 DEEP:          {deep:>3} ({deep/n*100:.0f}%)")
        print(f"  60-79  PARTIAL DEEP:  {partial:>3} ({partial/n*100:.0f}%)")
        print(f"  40-59  SURFACE:       {surface:>3} ({surface/n*100:.0f}%)")
        print(f"  20-39  GENERIC:       {generic:>3} ({generic/n*100:.0f}%)")
        print(f"   0-19  WRONG:         {wrong:>3} ({wrong/n*100:.0f}%)")

    print(f"\n--- Top 15 ---")
    for r in char_results[:15]:
        print(f"  {r['avg_score']:>5.0f}  #{r['id']:>3} {r['character']:<25}")
        for s in r["scores"][:2]:
            print(f"         [{s['score']:>3}] {s['reason'][:60]}")
            print(f"               → {s['reframe'][:70]}")

    print(f"\n--- Bottom 15 ---")
    for r in char_results[-15:]:
        print(f"  {r['avg_score']:>5.0f}  #{r['id']:>3} {r['character']:<25}")
        for s in r["scores"][:1]:
            print(f"         [{s['score']:>3}] {s['reason'][:60]}")

    output = Path(__file__).resolve().parent.parent / "data" / "literary_gt_llm_judge.json"
    with open(output, "w") as f:
        json.dump(char_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
