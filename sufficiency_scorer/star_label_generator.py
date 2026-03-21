"""LLM-powered star label generation — V2 with precomputed signals.

Takes detector signal + precomputed context + user text → 3-6 char Chinese label.
Philosophy: 王阳明"致良知" — every person has inner light.
"""

import os

from sufficiency_scorer.precompute import precompute, format_precomputed

# Core insight from DBT + best-performing labels analysis:
# The best labels NAME A PARADOX the person is living in.
# "被爱淹没的孤独" = love + loneliness (82分)
# "微笑下的沸腾" = smile + boiling (78分)
# The worst labels describe one thing: "向内看的人" (35分)

SYSTEM = """Output a 4-8 character Chinese label for this person's soul map star.

TWO modes depending on tone:

PAIN/STRUGGLE/CONFLICT → name the PARADOX (two opposing forces, NOT the scene):
  "微笑下的沸腾" "爱里的筋疲力尽" "两杯咖啡的习惯" "擅长里的陌生" "坚强里的土崩"
  NEVER describe the scene (where/when). Always name the emotional tension.

JOY/GRATITUDE/CURIOSITY/EXCITEMENT → name the ESSENCE (celebrate the light):
  "三年磨出的光" "抱紧每一天" "打开所有门的人" "好奇不灭"

Choose mode based on what the person actually said. Don't force darkness onto light.
BANNED: generic labels like "内心柔软" "向内看的人" "敢于面对"

Output ONLY the Chinese label. Nothing else."""

USER = """{dimension}/{signal_key} | {context}
"{text}"
"""


class StarLabelGenerator:
    """Generates personalized star labels via LLM with precomputed context."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None
        if self._api_key.startswith("sk-or-"):
            self._model = "anthropic/claude-haiku-4-5-20251001"
            self._base_url = "https://openrouter.ai/api"
        else:
            self._model = "claude-haiku-4-5-20251001"
            self._base_url = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def generate_label(self, dimension: str, signal_key: str, user_text: str) -> str | None:
        """Generate one personalized star label with precomputed context."""
        try:
            # Precompute context signals (0ms)
            pc = precompute(user_text)
            context = format_precomputed(pc)

            client = self._get_client()
            r = client.messages.create(
                model=self._model,
                max_tokens=20,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER.format(
                    dimension=dimension,
                    signal_key=signal_key,
                    context=context,
                    text=user_text[:150],
                )}],
            )
            label = r.content[0].text.strip().strip('"\'「」【】。，')
            # Remove any trailing punctuation or explanation
            label = label.split('\n')[0].split('。')[0].split('，')[0].strip()
            if 2 <= len(label) <= 12:
                return label
            return None
        except Exception:
            return None
