"""Star label generator — minimal prompt, precomputed signals."""

import os
from sufficiency_scorer.precompute import precompute, format_precomputed

# 55 tokens. Pain→paradox, joy→essence. Examples do the heavy lifting.
SYSTEM = """Output one 4-8 char Chinese star label. Pain→paradox(A中的B), Joy→essence(celebrate).
Ex: "微笑下的沸腾" "两杯咖啡的习惯" "三年磨出的光" "擅长里的陌生" "抱紧每一天"
NO generic: "内心柔软" "向内看" "敢于面对". Label only, nothing else."""

USER = """{dim}/{key}|{ctx}
"{txt}"
"""


class StarLabelGenerator:
    """One Haiku call per star. ~55 system + ~40 user tokens."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None
        if self._api_key.startswith("sk-or-"):
            self._model = "anthropic/claude-haiku-4-5-20251001"
            self._base_url = "https://openrouter.ai/api"
        else:
            self._model = "claude-haiku-4-5-20251001"
            self._base_url = None
        self._pc_cache: dict[str, str] = {}

    def _get_client(self):
        if self._client is None:
            import anthropic
            kw = {"api_key": self._api_key}
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kw)
        return self._client

    def generate_label(self, dimension: str, signal_key: str, user_text: str) -> str | None:
        """Generate one star label. Caches precompute per text."""
        try:
            # Cache precompute (same text = same context)
            ctx = self._pc_cache.get(user_text)
            if ctx is None:
                ctx = format_precomputed(precompute(user_text))
                self._pc_cache[user_text] = ctx

            r = self._get_client().messages.create(
                model=self._model,
                max_tokens=15,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER.format(
                    dim=dimension, key=signal_key, ctx=ctx, txt=user_text[:120],
                )}],
            )
            label = r.content[0].text.strip().strip('"\'「」【】。，:：')
            label = label.split('\n')[0].split('。')[0].split('，')[0].strip()
            return label if 2 <= len(label) <= 12 else None
        except Exception:
            return None
