"""V3 insight generator — minimal prompt + precomputed signals.

Design principles:
1. Precompute everything possible (topics, tone, key phrases) — zero cost
2. Send LLM the MINIMUM it needs: precomputed signals + user text + short instruction
3. Constrain output format to reduce variability
4. Target: <80 tokens system prompt, <60 tokens user prompt template
"""

import json
import os

from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)
from sufficiency_scorer.precompute import precompute, format_precomputed

# 47 tokens — every word earns its place
SYSTEM = """Generate 3 personality insights from someone's words + analysis signals.
Rules: Quote their words. Reveal hidden meaning. No generic praise. Match their tone. JSON array only."""

# ~30 tokens template (excluding variable content)
USER = """Signals: {signals}
Text: "{text}"
[{{"s":"signal","i":"insight"}},...]"""


class InsightGeneratorV3:
    """Minimal-prompt insight generator. ~80 input tokens + user text."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
            if self._api_key.startswith("sk-or-"):
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    base_url="https://openrouter.ai/api",
                )
            else:
                self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, user_text: str) -> list[InsightCandidate]:
        """Generate insights from user text. Precomputes signals internally."""
        pc = precompute(user_text)
        signals = format_precomputed(pc)

        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": USER.format(
                signals=signals, text=user_text[:300],
            )}],
        )
        return self._parse(response.content[0].text.strip())

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
            reframe = item.get("i", item.get("insight", item.get("text", "")))
            source = item.get("s", item.get("source", item.get("signal", "")))
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
