"""Star Engine PDCA — simulate 100 natural users, measure, iterate.

Each iteration:
1. Run 100 personas through StarEngine (1 turn = their first message)
2. LLM-as-judge: persona scores their stars (0-100)
3. Analyze failures → identify improvement
4. Apply fix → next iteration

Uses Haiku for speed (both generation and judging).
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.literary_validation import (
    heuristic_emotion, heuristic_conflict, heuristic_humor, heuristic_fragility,
)
from sufficiency_scorer.detectors.eq import EQAdapter
from sufficiency_scorer.models import DetectorResult, Dimension
from sufficiency_scorer.star_engine import StarEngine
from sufficiency_scorer.star_label_generator import StarLabelGenerator

DATA = Path(__file__).resolve().parent.parent / "data" / "natural_personas.json"

import anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(
    api_key=api_key,
    base_url="https://openrouter.ai/api" if api_key.startswith("sk-or-") else "https://api.anthropic.com",
)
MODEL = "anthropic/claude-haiku-4-5-20251001" if api_key.startswith("sk-or-") else "claude-haiku-4-5-20251001"


def judge_stars(persona_name: str, persona_text: str, labels: list[str]) -> dict:
    """LLM judge: how would this persona feel about these star labels?"""
    prompt = f"""You are a person who just said this to an app:
"{persona_text}"

The app analyzed your words and showed you these stars on a soul map:
{chr(10).join("★ " + l for l in labels)}

Score 0-100 how this makes you feel:
- 90-100: "Wow, this app gets me. These stars feel like ME." Stunned and moved.
- 70-89: "Pretty accurate. I feel seen." Impressed.
- 50-69: "Some of these fit, some are generic." OK but not special.
- 30-49: "These could be anyone. Not personal." Disappointing.
- 0-29: "This is wrong. It doesn't understand me at all."

Consider: Do the labels collectively paint a picture of who you are? Or could they apply to anyone?

JSON only: {{"score": N, "best_label": "which star resonated most", "worst_label": "which felt most wrong or generic", "what_missing": "what should it have seen instead"}}"""

    r = client.messages.create(
        model=MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    resp = r.content[0].text.strip()
    if "```" in resp:
        resp = resp.split("\n", 1)[1].rsplit("```", 1)[0].strip() if "\n" in resp else resp
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        import re
        m = re.search(r'"score"\s*:\s*(\d+)', resp)
        return {"score": int(m.group(1)) if m else 40, "best_label": "", "worst_label": "", "what_missing": resp[:100]}


async def run_one_iteration(personas: list[dict], iteration: int) -> dict:
    """Run one full iteration: 100 users → stars → judge → analyze."""
    eq = EQAdapter()
    scores = []
    feedbacks = []
    label_counter = Counter()
    star_counts = []

    for p in personas:
        text = p["text"]
        # Detectors
        eq_r = await eq.detect(text)
        results = [heuristic_emotion(text), eq_r, heuristic_conflict(text),
                   heuristic_humor(text), heuristic_fragility(text)]
        seen = {r.dimension for r in results}
        for dim in Dimension:
            if dim not in seen:
                results.append(DetectorResult(dimension=dim))

        # Star Engine with LLM label generator
        label_gen = StarLabelGenerator()
        engine = StarEngine(label_generator=label_gen)
        output = engine.process_turn(results, turn_count=1, user_text=text)
        labels = [s.star.label for s in output.new_stars]

        # If too few stars, simulate turn 2
        if len(engine.stars) < 2:
            output2 = engine.process_turn(results, turn_count=2, user_text=text)
            labels += [s.star.label for s in output2.new_stars]

        for l in labels:
            label_counter[l] += 1
        star_counts.append(len(engine.stars))

        # Judge
        if labels:
            result = judge_stars(p["name"], text[:200], labels[:4])
            score = result.get("score", 40)
            scores.append(score)
            feedbacks.append({
                "id": p["id"],
                "name": p["name"],
                "spectrum": p["spectrum"],
                "score": score,
                "labels": labels[:4],
                "best": result.get("best_label", ""),
                "worst": result.get("worst_label", ""),
                "missing": result.get("what_missing", ""),
            })

    # Analyze
    mean_score = sum(scores) / len(scores) if scores else 0
    high = sum(1 for s in scores if s >= 70)
    low = sum(1 for s in scores if s < 50)

    return {
        "iteration": iteration,
        "mean_score": round(mean_score, 1),
        "high_pct": round(high / len(scores) * 100) if scores else 0,
        "low_pct": round(low / len(scores) * 100) if scores else 0,
        "avg_stars": round(sum(star_counts) / len(star_counts), 1) if star_counts else 0,
        "unique_labels": len(label_counter),
        "top_labels": label_counter.most_common(5),
        "feedbacks": feedbacks,
        "worst_5": sorted(feedbacks, key=lambda f: f["score"])[:5],
        "best_5": sorted(feedbacks, key=lambda f: -f["score"])[:5],
    }


async def main():
    with open(DATA) as f:
        personas = json.load(f)

    print(f"Loaded {len(personas)} natural personas")
    print("=" * 80)

    results = []
    for i in range(1, 2):  # 1 iteration (V3 paradox labels)
        print(f"\n--- ITERATION {i} ---")
        t0 = time.monotonic()
        r = await run_one_iteration(personas, i)
        elapsed = time.monotonic() - t0

        results.append(r)
        print(f"  Mean: {r['mean_score']}/100  ≥70: {r['high_pct']}%  <50: {r['low_pct']}%  Stars: {r['avg_stars']}  Labels: {r['unique_labels']}  Time: {elapsed:.0f}s")
        print(f"  Top labels: {', '.join(f'{l}({c})' for l, c in r['top_labels'])}")

        print(f"  Worst 5:")
        for f in r["worst_5"]:
            print(f"    [{f['score']:>3}] {f['name']:<15} {f['spectrum']:<15} labels={f['labels'][:2]}")
            if f["missing"]:
                print(f"         Missing: {f['missing'][:60]}")

        print(f"  Best 5:")
        for f in r["best_5"]:
            print(f"    [{f['score']:>3}] {f['name']:<15} {f['spectrum']:<15} labels={f['labels'][:2]}")

        # Analyze feedback for next iteration
        worst_spectrums = Counter(f["spectrum"] for f in r["feedbacks"] if f["score"] < 50)
        if worst_spectrums:
            print(f"  Problem spectrums: {dict(worst_spectrums.most_common(5))}")

        missing_themes = Counter()
        for f in r["feedbacks"]:
            if f["score"] < 60 and f["missing"]:
                for word in f["missing"].lower().split():
                    if len(word) > 4 and word not in {"about", "these", "would", "should", "could", "their", "which", "other", "being", "where", "there", "label", "labels", "stars"}:
                        missing_themes[word] += 1
        if missing_themes:
            print(f"  Missing themes: {dict(missing_themes.most_common(10))}")

        # Note: In a real PDCA, we'd modify star_labels.py here based on feedback
        # For now, each iteration uses the same engine (measuring variance)

    # Final summary
    print("\n" + "=" * 80)
    print("PDCA PROGRESSION")
    print("=" * 80)
    for r in results:
        bar = "█" * int(r["mean_score"] // 5)
        print(f"  R{r['iteration']:>2}: {r['mean_score']:>5.1f}/100 {bar}  ≥70={r['high_pct']:>3}%  <50={r['low_pct']:>3}%")

    output = Path(__file__).resolve().parent.parent / "data" / "star_pdca_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
