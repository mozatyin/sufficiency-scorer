"""Star label generator — minimal prompt, precomputed signals, multi-language."""

import os
from sufficiency_scorer.precompute import precompute, format_precomputed

# Per-language system prompts. Examples in target language do the heavy lifting.
SYSTEM_PROMPTS = {
    "zh": """Output one 4-8 char Chinese star label. Pain→paradox(A中的B), Joy→essence(celebrate).
Ex: "微笑下的沸腾" "两杯咖啡的习惯" "三年磨出的光" "擅长里的陌生" "抱紧每一天"
NO generic: "内心柔软" "向内看" "敢于面对". Label only, nothing else.""",

    "en": """Output one 2-5 word English star label. Pain→paradox(A within B), Joy→essence(celebrate).
Ex: "Smile hiding fire" "Two cups of coffee" "Three years to light" "Skilled yet hollow" "Holding every day"
NO generic: "Self-aware" "Strong inside" "Brave". Label only, nothing else.""",

    "ar": """أنشئ عنوان نجمة واحد من 2-5 كلمات بالعربية. ألم→مفارقة(أ في ب), فرح→جوهر(احتفال).
أمثلة: "ابتسامة تخفي حريقاً" "فنجانا قهوة بلا شريك" "ثلاث سنوات نحو النور" "بارع لكن فارغ"
ممنوع العام: "قوي من الداخل" "شجاع". العنوان فقط، بدون شرح.""",
}

DEFAULT_LANG = "zh"

USER = """{dim}/{key}|{ctx}
"{txt}"
"""


class StarLabelGenerator:
    """One Haiku call per star. Multi-language via per-language system prompt."""

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

    def generate_label(
        self, dimension: str, signal_key: str, user_text: str, language: str = "zh"
    ) -> str | None:
        """Generate one star label in the specified language."""
        try:
            ctx = self._pc_cache.get(user_text)
            if ctx is None:
                ctx = format_precomputed(precompute(user_text))
                self._pc_cache[user_text] = ctx

            system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[DEFAULT_LANG])

            r = self._get_client().messages.create(
                model=self._model,
                max_tokens=20,
                system=system,
                messages=[{"role": "user", "content": USER.format(
                    dim=dimension, key=signal_key, ctx=ctx, txt=user_text[:120],
                )}],
            )
            label = r.content[0].text.strip().strip('"\'「」【】。，:：')
            label = label.split('\n')[0].split('。')[0].split('，')[0].strip()
            # Length validation varies by language
            if language == "zh":
                return label if 2 <= len(label) <= 12 else None
            else:
                # en/ar: word count 2-6
                words = label.split()
                return label if 2 <= len(words) <= 8 else None
        except Exception:
            return None
