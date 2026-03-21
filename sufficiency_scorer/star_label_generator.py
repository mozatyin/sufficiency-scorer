"""LLM-powered star label generation — V2 with precomputed signals.

Takes detector signal + precomputed context + user text → 3-6 char Chinese label.
Philosophy: 王阳明"致良知" — every person has inner light.
"""

import os

from sufficiency_scorer.precompute import precompute, format_precomputed

# Compact system prompt — every token earns its place
SYSTEM = """你是 SoulMap 星标签生成器。从用户的话和分析信号中，生成一个 3-6 字的中文星标签。

核心原则（王阳明：我心光明）：
- 指向内心光明面，即使被乌云遮蔽
- 必须触及情感核心，不只是描述场景
- 必须基于用户说的具体内容

场景 vs 情感核心的区别：
✗ "淋浴里的真实" — 只描述了场景（在哪里哭）
✓ "不屈的温柔" — 触及了情感核心（每天被霸凌还能保持柔软）

✗ "肾上腺素的选择" — 浪漫化了行为表面
✓ "挣扎中的诚实" — 看到了承认问题的勇气

✗ "向内看的人" — 通用，谁都行
✓ "加班背后的底线" — 具体到这个人的处境

标签要让用户想："它怎么知道的？"而不是"嗯，还行吧"。"""

USER = """信号: {dimension} — {signal_key}
上下文: {context}
用户: "{text}"

一个标签（3-6字），只返回文字："""


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
            label = r.content[0].text.strip().strip('"\'「」【】')
            if 2 <= len(label) <= 10:
                return label
            return None
        except Exception:
            return None
