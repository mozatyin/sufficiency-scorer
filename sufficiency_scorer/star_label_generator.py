"""LLM-powered star label generation — personalized, not template.

Takes a detector signal + user's actual words → generates a 3-6 character
Chinese label that's specific to THIS user.

Philosophy: 王阳明"致良知" — every person has inner light. Labels point to
the light within, even when it's covered by clouds.
"""

import json
import os

SYSTEM = """你是 SoulMap 的星标签生成器。

用户说了一段话，AI 检测器发现了某个维度的信号。你要生成一个 3-6 个字的星标签。

规则：
1. 标签必须正面 — 指向用户内心的光明面（王阳明：我心光明）
2. 标签必须具体 — 基于用户说的具体内容，不能是通用语
3. 3-6 个中文字，可以加"?"表示暗示
4. 禁止临床术语、负面人格判断
5. 每个人的标签都应该不同

坏: "内心柔软"（太通用，谁都行）
好: "加班背后的底线"（具体引用了用户的情境）
好: "笑着扛一切"（基于用户自嘲的方式）"""

USER = """信号: {dimension} — {signal_key}
用户原话: "{text}"

生成1个标签（3-6字）。只返回标签文字，不要引号不要解释。"""


class StarLabelGenerator:
    """Generates personalized star labels via LLM. One call per star."""

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
        """Generate one personalized star label. Returns 3-6 char Chinese label."""
        try:
            client = self._get_client()
            r = client.messages.create(
                model=self._model,
                max_tokens=30,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER.format(
                    dimension=dimension,
                    signal_key=signal_key,
                    text=user_text[:150],
                )}],
            )
            label = r.content[0].text.strip().strip('"').strip("'").strip("「」")
            # Validate: 2-10 Chinese characters
            if 2 <= len(label) <= 10:
                return label
            return None
        except Exception:
            return None
