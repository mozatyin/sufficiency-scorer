"""Fast insight generation — single minimal LLM call.

Combines detection + generation into one call with minimal prompt.
Target: <2s latency with Haiku, same quality as Sonnet.

Architecture comparison:
  Standard: EQ(0ms) + heuristics(0ms) + Sonnet(6s) = 6s
  Fast:     EQ(0ms) + Haiku(3s) = 3s
  Ideal:    EQ(0ms) + Haiku(1.5s with streaming) = 1.5s
"""

import json
import os

from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)
from sufficiency_scorer.insight_generator import format_signals


FAST_SYSTEM = """You generate personality insights for SoulMap. You receive behavioral analysis signals + the user's exact words.

RULES:
1. Quote specific phrases from their text — reveal what those words REALLY mean
2. Each insight = "X isn't about X, it's about Y" pattern
3. At least one gut-punch (uncomfortably accurate)
4. NO generic Barnum statements. "You're self-aware" = BANNED.
5. Match their tone. Don't force positivity on dark/cold/sarcastic people.
6. 1-2 sentences each. JSON array only."""

FAST_USER = """BEHAVIORAL SIGNALS:
{signals}

THEIR WORDS:
"{text}"

Generate 3-4 insights. JSON array: [{{"source": "signal", "insight": "..."}}]"""


class FastInsightGenerator:
    """Single LLM call: detect + generate combined. ~3s with Haiku."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if model:
            self._model = model
        elif self._api_key.startswith("sk-or-"):
            self._model = "anthropic/claude-haiku-4-5-20251001"
        else:
            self._model = "claude-haiku-4-5-20251001"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
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
        signals = format_signals(results)
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=500,
            system=FAST_SYSTEM,
            messages=[{"role": "user", "content": FAST_USER.format(
                signals=signals, text=user_text,
            )}],
        )
        text = response.content[0].text.strip()
        return self._parse(text)

    def _parse(self, text: str) -> list[InsightCandidate]:
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        insights = []
        for item in items:
            if not isinstance(item, dict):
                continue
            reframe = item.get("insight", item.get("text", ""))
            source = item.get("source", item.get("signal_source", ""))
            if not reframe:
                continue
            insights.append(InsightCandidate(
                source_dimensions=[Dimension.EMOTION],
                signal=source,
                reframe=reframe,
                quality=InsightQuality.MEDIUM,
                confidence=0.7,
            ))
        return insights
