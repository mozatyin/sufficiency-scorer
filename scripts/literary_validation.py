"""Run 100 literary characters through the sufficiency scorer pipeline.

For each character's first-meeting dialogue:
1. Run EQ adapter (real, zero-cost)
2. Simulate other detectors from text heuristics
3. Run InsightExtractor with user_text
4. Run InsightEvaluator on results
5. Report: what dimensions activate, what insights emerge, quality scores
"""

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sufficiency_scorer.detectors.eq import EQAdapter, extract_behavioral, compute_valence, compute_distress
from sufficiency_scorer.insight_extractor import InsightExtractor
from sufficiency_scorer.evaluator import InsightEvaluator
from sufficiency_scorer.models import DetectorResult, Dimension, InsightQuality

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "literary_first_words.json"


# --- Text heuristics for simulating detectors ---

CONFLICT_KEYWORDS = {
    "fight", "argue", "conflict", "battle", "attack", "force", "forcing",
    "against", "enemy", "confront", "criticize", "blame", "accuse", "war",
    "struggle", "resist", "refuse", "defend", "betray", "injustice",
    "punish", "revenge", "threat", "hate", "kill", "murder", "hurt",
    "abuse", "violent", "cruelty", "unfair", "wrong", "broken",
    "shut in my face", "took from me", "they made", "sin against",
    "mortified", "never say anything", "can't say no", "trapped",
    "power", "control", "manipulate", "own them", "leverage",
    "nobody listens", "talking to a wall",
}

HUMOR_KEYWORDS = {
    "haha", "lol", "funny", "joke", "laugh", "hilarious", "comedy",
    "ironic", "sarcastic", "at least", "consistent right", "fool",
    "amuse", "ridiculous", "absurd",
}

FRAGILITY_OPEN_KEYWORDS = {
    "afraid", "terrified", "scared", "don't know", "can't",
    "lost", "help", "alone", "vulnerable", "honest", "crying",
    "broken", "dying", "death", "grief", "miss", "gone",
    "hurting", "suffering", "pain", "overwhelmed",
    "I need", "I want", "please", "I wish", "what happened",
    "I feel", "I'm not", "confused", "frightened", "heavy",
    "wrong with me", "terrifies me", "never get free",
    "deserved", "how to be normal", "nightmare",
}

FRAGILITY_DEFENSIVE_KEYWORDS = {
    "fine", "perfectly", "don't need", "I'm good", "whatever",
    "doesn't matter", "I don't care", "leave me alone",
    "I'm completely fine", "adequate", "I'm not scared",
    "smarter than", "own them", "easy to read",
}

FRAGILITY_MASKED_KEYWORDS = {
    "pretend", "performance", "surface", "mask", "hide",
    "nobody knows", "they don't see", "invisible",
    "act like", "performing", "underneath",
    "people see", "what they want to see", "smile",
    "carry more", "holding", "buried", "inside",
}

IDENTITY_KEYWORDS = {
    "who am I", "who I am", "identity", "belong", "purpose",
    "meaning", "figure out", "lost", "don't know myself",
    "name", "called", "real me",
}


def heuristic_conflict(text: str) -> DetectorResult:
    text_lower = text.lower()
    hits = sum(1 for k in CONFLICT_KEYWORDS if k in text_lower)
    if hits >= 2:
        # Determine dominant style from context
        if any(w in text_lower for w in ("quiet", "silence", "hold back", "don't say", "avoid", "never say")):
            style = "avoid"
        elif any(w in text_lower for w in ("fight", "confront", "face", "stand up", "backbone")):
            style = "confront"
        else:
            style = "avoid"  # default for internal conflict
        conf = min(0.3 + hits * 0.1, 0.9)
        return DetectorResult(
            dimension=Dimension.CONFLICT, activated=True, confidence=conf,
            detail={"styles": {style: conf, "compromise": 0.2}},
        )
    return DetectorResult(dimension=Dimension.CONFLICT)


def heuristic_humor(text: str) -> DetectorResult:
    text_lower = text.lower()
    hits = sum(1 for k in HUMOR_KEYWORDS if k in text_lower)
    if hits >= 1:
        # Determine style
        if any(w in text_lower for w in ("myself", "I'm", "my fault", "common denominator", "at least I")):
            style = "self_deprecating"
        elif any(w in text_lower for w in ("boys", "game", "together", "lively")):
            style = "affiliative"
        else:
            style = "self_enhancing"
        conf = min(0.3 + hits * 0.15, 0.9)
        return DetectorResult(
            dimension=Dimension.HUMOR, activated=True, confidence=conf,
            detail={"humor_detected": True, "styles": {style: conf}},
        )
    return DetectorResult(dimension=Dimension.HUMOR)


def heuristic_fragility(text: str) -> DetectorResult:
    text_lower = text.lower()
    open_hits = sum(1 for k in FRAGILITY_OPEN_KEYWORDS if k in text_lower)
    defensive_hits = sum(1 for k in FRAGILITY_DEFENSIVE_KEYWORDS if k in text_lower)
    masked_hits = sum(1 for k in FRAGILITY_MASKED_KEYWORDS if k in text_lower)

    max_hits = max(open_hits, defensive_hits, masked_hits)
    if max_hits >= 1:
        if open_hits >= defensive_hits and open_hits >= masked_hits:
            pattern = "open"
            conf = min(0.3 + open_hits * 0.08, 0.9)
        elif masked_hits >= defensive_hits:
            pattern = "masked"
            conf = min(0.3 + masked_hits * 0.1, 0.9)
        else:
            pattern = "defensive"
            conf = min(0.3 + defensive_hits * 0.1, 0.9)
        return DetectorResult(
            dimension=Dimension.FRAGILITY, activated=True, confidence=conf,
            detail={"pattern": pattern, "pattern_scores": {pattern: conf}},
        )
    return DetectorResult(dimension=Dimension.FRAGILITY)


def heuristic_emotion(text: str) -> DetectorResult:
    """Simulate emotion detection from text keywords."""
    text_lower = text.lower()
    emotions = {}

    # Map keywords to emotions — expanded for literary dialogue
    emotion_keywords = {
        "frustration": ["tired", "can't stand", "sick of", "fed up", "suffocating", "trapped",
                         "exhausted", "stuck", "going nowhere", "can't take", "unbearable",
                         "won't let me", "forcing", "pushed", "no escape", "pointless"],
        "anger": ["hate", "furious", "angry", "unfair", "injustice", "wrong", "cruelty",
                   "punish", "revenge", "destroy", "sin against", "they took", "stolen",
                   "mortified", "how dare", "how can"],
        "sadness": ["sad", "miss", "gone", "lost", "cry", "tears", "mourn", "grief",
                     "died", "passed away", "death", "lonely", "alone", "nobody",
                     "empty", "flat", "weight", "heavy", "ache", "hurts", "breaking"],
        "fear": ["afraid", "terrified", "scared", "frightened", "panic", "horror",
                  "dread", "danger", "kill me", "they'll find", "what happens",
                  "not safe", "protect", "threat"],
        "anxiety": ["worry", "anxious", "can't sleep", "racing", "nervous", "tense",
                     "restless", "flutter", "slipping", "what if", "might",
                     "keep going back", "can't stop thinking", "obsess"],
        "confusion": ["don't understand", "don't know", "confused", "no clue",
                       "what's happening", "makes no sense", "who am I", "figure out",
                       "uncertain", "question", "wonder", "why"],
        "hope": ["believe", "faith", "maybe", "one day", "possible", "dream",
                  "wish", "better", "discover", "new", "start over", "chance",
                  "learn", "grow", "ready"],
        "despair": ["hopeless", "never", "nothing", "pointless", "meaningless",
                     "empty", "no way out", "drowning", "darkness", "void",
                     "whole life has been wrong", "never truly lived", "all there is"],
        "grief": ["passed away", "died", "death", "funeral", "mourning", "loss",
                   "gone forever", "buried", "grave", "he's gone", "she's gone",
                   "never coming back"],
        "love": ["love", "loved", "heart", "dear", "care deeply", "connection",
                  "warmth", "devoted", "cherish", "miss you", "always"],
        "guilt": ["fault", "blame myself", "should have", "regret", "sorry",
                   "ashamed", "mistake", "I destroyed", "what I did", "undo",
                   "atonement", "wrong thing"],
        "shame": ["ashamed", "embarrassed", "humiliated", "marked", "stain",
                   "illegitimate", "ugly", "invisible", "refuse to see",
                   "nobody thinks", "worthless"],
        "determination": ["I will", "I must", "I'm going to", "no matter what",
                           "I choose", "I volunteer", "I'm not afraid", "I'll go",
                           "fight for", "ready to", "choosing to", "I can"],
        "amusement": ["haha", "funny", "laugh", "hilarious", "amuse",
                       "comedy", "joke", "at least"],
        "loneliness": ["alone", "nobody", "invisible", "no one", "by myself",
                        "strangers", "performing", "real me", "isolated",
                        "disconnected", "no friend"],
        "irritation": ["annoying", "bother", "nag", "tiresome", "can't stand",
                        "ridiculous", "nonsense", "disaster", "what's wrong with"],
    }

    for emotion, keywords in emotion_keywords.items():
        hits = sum(1 for k in keywords if k in text_lower)
        if hits > 0:
            emotions[emotion] = min(0.2 + hits * 0.15, 0.95)

    if not emotions:
        return DetectorResult(dimension=Dimension.EMOTION)

    top = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
    max_conf = top[0][1]
    return DetectorResult(
        dimension=Dimension.EMOTION, activated=True, confidence=max_conf,
        detail={"top_emotions": top},
    )


async def analyze_character(char: dict, eq_adapter: EQAdapter, extractor: InsightExtractor, evaluator: InsightEvaluator) -> dict:
    """Analyze one character's first words."""
    text = char["text"]

    # Real EQ
    eq_result = await eq_adapter.detect(text)

    # Heuristic detectors
    emotion_result = heuristic_emotion(text)
    conflict_result = heuristic_conflict(text)
    humor_result = heuristic_humor(text)
    fragility_result = heuristic_fragility(text)

    # Assemble results
    results = [emotion_result, eq_result, conflict_result, humor_result, fragility_result]
    seen = {r.dimension for r in results}
    for dim in Dimension:
        if dim not in seen:
            results.append(DetectorResult(dimension=dim))

    # Extract insights with dynamic reframing
    insights = extractor.extract(results, user_text=text)
    good = [i for i in insights if i.quality.value >= InsightQuality.MEDIUM.value]

    # Evaluate quality
    batch_eval = evaluator.evaluate_batch(insights)

    activated_dims = [r.dimension.value for r in results if r.activated]

    return {
        "id": char["id"],
        "character": char["character"],
        "novel": char["novel"],
        "words": len(text.split()),
        "activated": activated_dims,
        "activated_count": len(activated_dims),
        "total_insights": len(insights),
        "good_insights": len(good),
        "ready": len(good) >= 3,
        "avg_specificity": batch_eval.avg_specificity,
        "avg_reframe": batch_eval.avg_reframe_quality,
        "bloom_worthy": batch_eval.bloom_worthy,
        "insights": [
            {
                "quality": i.quality.name,
                "signal": i.signal[:60],
                "reframe": i.reframe[:100],
                "dims": [d.value for d in i.source_dimensions],
            }
            for i in insights
        ],
    }


async def main():
    with open(DATA_PATH) as f:
        characters = json.load(f)

    print(f"Loaded {len(characters)} characters\n")

    eq = EQAdapter()
    extractor = InsightExtractor()
    evaluator = InsightEvaluator()

    results = []
    for char in characters:
        r = await analyze_character(char, eq, extractor, evaluator)
        results.append(r)

    # === Summary Statistics ===
    print("=" * 80)
    print("LITERARY VALIDATION: 100 Characters Through Sufficiency Scorer")
    print("=" * 80)

    ready_count = sum(1 for r in results if r["ready"])
    bloom_count = sum(1 for r in results if r["bloom_worthy"])
    word_counts = [r["words"] for r in results]

    print(f"\n--- Overview ---")
    print(f"  Characters: {len(results)}")
    print(f"  Word range: {min(word_counts)}-{max(word_counts)} (mean {sum(word_counts)/len(word_counts):.0f})")
    print(f"  Ready (≥3 MEDIUM+): {ready_count}/{len(results)} ({ready_count/len(results)*100:.0f}%)")
    print(f"  Bloom-worthy (eval): {bloom_count}/{len(results)} ({bloom_count/len(results)*100:.0f}%)")

    # Dimension activation frequency
    dim_counts = Counter()
    for r in results:
        for d in r["activated"]:
            dim_counts[d] += 1

    print(f"\n--- Dimension Activation Frequency ---")
    for dim, count in dim_counts.most_common():
        bar = "█" * (count // 2)
        print(f"  {dim:<12} {count:>3}/100 {bar}")

    # Insight quality distribution
    all_qualities = Counter()
    for r in results:
        for i in r["insights"]:
            all_qualities[i["quality"]] += 1

    print(f"\n--- Insight Quality Distribution ---")
    total_insights = sum(all_qualities.values())
    for q in ["HIGH", "MEDIUM", "LOW", "NOISE"]:
        count = all_qualities.get(q, 0)
        pct = count / total_insights * 100 if total_insights > 0 else 0
        print(f"  {q:<8} {count:>4} ({pct:.0f}%)")

    # Average quality scores
    specs = [r["avg_specificity"] for r in results if r["total_insights"] > 0]
    reframes = [r["avg_reframe"] for r in results if r["total_insights"] > 0]
    print(f"\n--- Quality Scores (among characters with insights) ---")
    print(f"  Avg specificity:    {sum(specs)/len(specs):.3f}")
    print(f"  Avg reframe quality: {sum(reframes)/len(reframes):.3f}")

    # Characters that DON'T reach threshold
    not_ready = [r for r in results if not r["ready"]]
    print(f"\n--- Characters NOT Ready ({len(not_ready)}) ---")
    for r in not_ready:
        print(f"  #{r['id']:>3} {r['character']:<25} [{r['words']}w] activated={r['activated']} insights={r['good_insights']}")

    # Top 10 richest characters (most insights)
    by_insights = sorted(results, key=lambda r: (r["good_insights"], r["avg_reframe"]), reverse=True)
    print(f"\n--- Top 10 Richest Characters ---")
    for r in by_insights[:10]:
        print(f"  #{r['id']:>3} {r['character']:<25} [{r['words']}w] {r['good_insights']} good insights, reframe={r['avg_reframe']:.2f}")
        for i in r["insights"][:3]:
            print(f"       [{i['quality']}] {i['reframe'][:75]}")

    # Bottom 10 (weakest)
    print(f"\n--- Bottom 10 (Weakest) ---")
    for r in by_insights[-10:]:
        print(f"  #{r['id']:>3} {r['character']:<25} [{r['words']}w] {r['good_insights']} good, activated={r['activated']}")

    # Save full results
    output_path = Path(__file__).resolve().parent.parent / "data" / "literary_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
