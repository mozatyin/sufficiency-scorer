"""LLM-powered insight generation — the final packaging step.

Takes detector signals (the "AlphaGo" analysis) + user's original text,
and generates personalized reframes via one LLM call.

Architecture:
  1. InsightExtractor identifies WHICH signals to use (rule-based, fast)
  2. InsightGenerator takes those signals + user text → personalized reframes (LLM, ~1s)

The LLM is NOT doing the analysis — that's the detectors' job.
The LLM is packaging the analysis into language that feels personal and surprising.
"""

import json
import os

from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)


SYSTEM_PROMPT = """You are a personality insight packager for SoulMap — an app that helps people feel deeply understood.

Your job: Take detector analysis results and the user's original words, then craft 3-5 personalized insights that make the user feel deeply seen.

## CORE PRINCIPLES

1. SPECIFICITY IS EVERYTHING
   - Every insight MUST quote or paraphrase a specific phrase from the user's text
   - "You're deeply self-aware" = BANNED (generic Barnum statement)
   - "When you say 'I don't want to do the OT anymore,' the word 'anymore' reveals this isn't sudden — it's been building" = GOOD (hyper-specific)

2. TRANSFORM, DON'T REPEAT
   - Take what they said and reveal a HIDDEN layer they didn't articulate
   - The user should think "I said that but I didn't realize THAT'S what I meant"
   - Use the "X is not about X, it's about Y" pattern:
     "Your exhaustion isn't about the hours — it's about doing something that stopped meaning anything to you"

3. MATCH THE PERSON'S TONE
   - If they're sarcastic/dark/cynical, DON'T force positivity. Mirror their intelligence.
   - If they sound manipulative or cold, observe their STRATEGY accurately instead of inventing warmth
   - "You read people faster than they read you" is better than "You care deeply" for a guarded person
   - Not everyone needs comfort. Some people need to be SEEN accurately.

4. ONE STUNNING INSIGHT > THREE SAFE ONES
   - At least one insight must be a "gut punch" — something uncomfortably accurate
   - The contrast pattern works: "You say X, but what you actually mean is Y"
   - Name the contradiction they're living in

5. GROUND EVERY INSIGHT IN A DETECTOR SIGNAL
   - Never invent analysis — only package what the detectors found
   - Cross-signal insights (combining 2+ detectors) are the most powerful

## FORMAT RULES
- Write in second person ("You...")
- 1-2 sentences per insight
- No clinical terms (depression, anxiety disorder, PTSD)
- No generic affirmations ("You're strong", "You're brave", "You're self-aware")

## ANTI-PATTERNS (instant quality failure)
- "You're deeply self-aware" — Barnum statement, applies to everyone
- "You feel [emotion name]" — just repeating the detector
- "You care deeply about [vague thing]" — generic positivity
- "You're not just [what they said]" followed by generic reframe — lazy pattern
- Forced positivity for someone who is clearly dark/manipulative/narcissistic"""


USER_PROMPT_TEMPLATE = """ORIGINAL TEXT (the user's exact words):
"{user_text}"

DETECTOR SIGNALS (what our analysis found):
{signals_json}

Generate exactly 4 insights. Requirements:
- Each MUST quote or reference a specific phrase from the original text
- Each MUST be grounded in a detector signal
- At least one must be a "gut punch" — uncomfortably accurate
- At least one must cross two detector signals
- Match the person's actual tone (don't force positivity if they're dark/cold/sarcastic)

JSON array only:
[
  {{"signal_source": "emotion + conflict", "insight": "Your insight here"}},
  ...
]"""


def format_signals(results: list[DetectorResult]) -> str:
    """Format activated detector results into a readable signal summary."""
    signals = []
    for r in results:
        if not r.activated:
            continue
        sig = {"dimension": r.dimension.value, "confidence": round(r.confidence, 2)}

        if r.dimension == Dimension.EMOTION:
            top = r.detail.get("top_emotions", [])[:3]
            sig["top_emotions"] = [{"name": n, "intensity": round(s, 2)} for n, s in top]

        elif r.dimension == Dimension.CONFLICT:
            styles = r.detail.get("styles", {})
            sig["conflict_styles"] = {k: round(v, 2) for k, v in sorted(styles.items(), key=lambda x: -x[1])[:2]}

        elif r.dimension == Dimension.HUMOR:
            styles = r.detail.get("styles", {})
            sig["humor_detected"] = r.detail.get("humor_detected", False)
            sig["humor_styles"] = {k: round(v, 2) for k, v in sorted(styles.items(), key=lambda x: -x[1])[:2]}

        elif r.dimension == Dimension.FRAGILITY:
            sig["pattern"] = r.detail.get("pattern")

        elif r.dimension == Dimension.EQ:
            features = r.detail.get("features", {})
            sig["behavioral"] = {
                "self_reference": round(features.get("self_ref", 0), 3),
                "question_ratio": round(features.get("question_ratio", 0), 3),
                "word_count": features.get("words", 0),
            }
            sig["valence"] = round(r.detail.get("valence", 0), 3)
            sig["distress"] = round(r.detail.get("distress", 0), 3)

        elif r.dimension == Dimension.SOULGRAPH:
            sig["intention_items"] = r.detail.get("items", 0)
            sig["avg_specificity"] = round(r.detail.get("avg_specificity", 0), 2)

        signals.append(sig)
    return json.dumps(signals, indent=2)


class InsightGenerator:
    """Generates personalized insights via one LLM call.

    Uses anthropic SDK. Requires ANTHROPIC_API_KEY env var.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Auto-detect model based on API key type
        if model:
            self._model = model
        elif self._api_key.startswith("sk-or-"):
            self._model = "anthropic/claude-sonnet-4"
        else:
            self._model = "claude-sonnet-4-20250514"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            # Support OpenRouter keys
            if self._api_key.startswith("sk-or-"):
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    base_url="https://openrouter.ai/api",
                )
            else:
                self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(
        self,
        results: list[DetectorResult],
        user_text: str,
    ) -> list[InsightCandidate]:
        """Generate personalized insights from detector signals + user text.

        Returns InsightCandidates with LLM-generated reframes.
        """
        signals_json = format_signals(results)
        activated = [r for r in results if r.activated]
        if not activated:
            return []

        prompt = USER_PROMPT_TEMPLATE.format(
            user_text=user_text,
            signals_json=signals_json,
        )

        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        text = response.content[0].text.strip()
        return self._parse_response(text, activated)

    def _parse_response(
        self, text: str, activated: list[DetectorResult]
    ) -> list[InsightCandidate]:
        """Parse LLM JSON response into InsightCandidates."""
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(items, list):
            return []

        # Map signal_source to dimensions
        dim_map = {r.dimension.value: r.dimension for r in activated}
        insights = []
        for item in items:
            if not isinstance(item, dict):
                continue
            reframe = item.get("insight", "")
            source = item.get("signal_source", "")
            if not reframe:
                continue

            # Determine source dimensions
            source_dims = []
            for dim_name, dim in dim_map.items():
                if dim_name in source.lower():
                    source_dims.append(dim)
            if not source_dims:
                source_dims = [activated[0].dimension]

            # Quality: cross-dimensional = HIGH, single = MEDIUM
            quality = InsightQuality.HIGH if len(source_dims) > 1 else InsightQuality.MEDIUM

            insights.append(InsightCandidate(
                source_dimensions=source_dims,
                signal=source,
                reframe=reframe,
                quality=quality,
                confidence=0.7,
            ))

        return insights
