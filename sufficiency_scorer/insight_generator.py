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


SYSTEM_PROMPT = """You are a personality insight packager for SoulMap.

Your job: Take detector analysis results and the user's original words, then craft 3-5 personalized insights that make the user feel deeply understood.

RULES:
1. Each insight must be grounded in a specific detector finding — never invent analysis
2. Reframe positively — turn every finding into a strength or meaningful observation
3. Reference the user's actual words/situation — never be generic
4. Never repeat the same insight twice
5. Never use clinical terms (depression, anxiety disorder, PTSD)
6. Never just repeat what the user said — transform it
7. Each insight should make the user think "how did you know that about me?"
8. Write in second person ("You...")
9. Keep each insight to 1-2 sentences
10. The insight should reveal something the user feels but hasn't articulated

BAD: "You're deeply self-aware" (generic, could apply to anyone)
BAD: "You feel frustrated" (just repeating the detector finding)
GOOD: "You're not just tired of overtime — you're realizing your boss's demands conflict with something you value more deeply" (specific, reframed, surprising)
GOOD: "The way you laugh at yourself after getting dumped isn't deflection — it's your way of staying honest when most people would hide" (cross-signal, personal)"""


USER_PROMPT_TEMPLATE = """Here is what our detectors found about this person, plus their original words.

ORIGINAL TEXT:
"{user_text}"

DETECTOR SIGNALS:
{signals_json}

Generate 3-5 personalized insights. Each insight must reference something specific from the original text AND be grounded in a detector signal. Format as JSON array:
[
  {{"signal_source": "emotion + conflict", "insight": "Your reframed insight here"}},
  ...
]

Return ONLY the JSON array, no other text."""


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
