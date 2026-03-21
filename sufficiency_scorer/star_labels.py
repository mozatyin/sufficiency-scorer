"""30/70 star label mapping — Gap 5.

70% positive frames: turn detector signals into strengths
30% dark stars (with ?): deeper probes the user can explore
Absolute ban: clinical terms, negative personality judgments, numeric scores
"""

from sufficiency_scorer.models import Dimension


# === 70% Positive Labels (亮星) ===
# dimension → signal_key → label

POSITIVE_LABELS: dict[Dimension, dict[str, str]] = {
    Dimension.EMOTION: {
        "frustration": "对不公平很敏感",
        "anger": "内心有火焰",
        "sadness": "内心柔软",
        "fear": "直觉很敏锐",
        "anxiety": "对风险有预感力",
        "confusion": "允许自己不确定",
        "hope": "内心有光",
        "despair": "承受过重量",
        "grief": "爱得很深",
        "happiness": "内心有光",
        "love": "天生的连接者",
        "guilt": "对自己要求很高",
        "shame": "在乎成为更好的人",
        "loneliness": "渴望深度连接",
        "irritation": "标准比别人高",
        "determination": "决定了就不放手",
        "amusement": "能在暗处找到光",
    },
    Dimension.CONFLICT: {
        "avoid": "先思考再行动",
        "confront": "直面问题的勇气",
        "compromise": "天生的调解者",
        "collaborate": "相信双赢",
        "compete": "为信念而战",
        "accommodating": "先想到别人",
    },
    Dimension.FRAGILITY: {
        "open": "敢于面对真实",
        "defensive": "用坚强保护柔软",
        "masked": "外表平静内心波涛",
        "denial": "用力量撑住一切",
    },
    Dimension.HUMOR: {
        "self_deprecating": "自嘲是你的超能力",
        "affiliative": "笑声是你的社交魔法",
        "aggressive": "看穿伪装的眼睛",
        "self_enhancing": "在风暴中找到幽默",
    },
    Dimension.MBTI: {
        "high_I": "内心世界丰富",
        "high_E": "从人群中获得能量",
        "high_T": "逻辑是你的武器",
        "high_F": "用心感受世界",
        "high_S": "脚踏实地的观察者",
        "high_N": "看到别人看不到的可能",
        "high_J": "掌控感让你安心",
        "high_P": "拥抱不确定性",
    },
    Dimension.LOVE_LANGUAGE: {
        "words": "渴望被肯定",
        "service": "行动胜过千言",
        "gifts": "重视心意的表达",
        "time": "在乎专属的陪伴",
        "touch": "需要温暖的连接",
    },
    Dimension.EQ: {
        "high_self_ref": "向内看的人",
        "high_question_ratio": "主动寻找出路",
        "high_distress": "正在经历重要的转变",
        "negative_valence": "在低谷中前行",
        "high_self_ref+question": "一边感受一边思考",
        "high_self_ref+distress": "清醒地承受着",
        "low_self_ref": "关注外面多过自己",
    },
    Dimension.CONNECTION_RESPONSE: {
        "turning_toward": "善于接住情感",
        "turning_away": "需要更多空间",
        "turning_against": "用距离保护自己",
    },
    Dimension.COMMUNICATION_DNA: {
        "verbose": "表达欲旺盛",
        "concise": "字字珠玑",
        "narrative": "天生的讲故事者",
        "analytical": "逻辑思维清晰",
    },
    Dimension.SOULGRAPH: {
        "clear_intention": "知道自己要什么",
        "exploring": "正在寻找方向",
        "conflicted": "内心在拔河",
    },
    Dimension.CHARACTER: {
        "resilient": "经历过风雨",
        "empathic": "能感受别人的感受",
        "independent": "习惯自己扛",
    },
}

# === 30% Dark Stars (暗星带问号) ===
# These are triggered by cross-dimensional patterns, not single signals.
# Each has a condition: (dim_a, signal_a, dim_b, signal_b) → dark label

DARK_LABELS: list[dict] = [
    {
        "condition": lambda results: _has(results, Dimension.CONFLICT, "avoid") and _has(results, Dimension.FRAGILITY, "defensive"),
        "label": "需要属于自己的空间?",
    },
    {
        "condition": lambda results: _has(results, Dimension.EMOTION, "frustration") and _has(results, Dimension.EQ, "high_question_ratio"),
        "label": "内心渴望改变?",
    },
    {
        "condition": lambda results: _has(results, Dimension.EMOTION, "sadness") and _has(results, Dimension.FRAGILITY, "masked"),
        "label": "藏起来的那部分想被看见?",
    },
    {
        "condition": lambda results: _has(results, Dimension.EMOTION, "loneliness") and _has(results, Dimension.LOVE_LANGUAGE, "time"),
        "label": "一直在等一个懂的人?",
    },
    {
        "condition": lambda results: _has(results, Dimension.FRAGILITY, "open") and _has(results, Dimension.EQ, "high_distress"),
        "label": "准备好放下什么了?",
    },
    {
        "condition": lambda results: _has(results, Dimension.EMOTION, "guilt") and _has(results, Dimension.CONFLICT, "avoid"),
        "label": "有些话一直没说出口?",
    },
    {
        "condition": lambda results: _has(results, Dimension.MBTI, "high_I") and _has(results, Dimension.FRAGILITY, "defensive"),
        "label": "内心的声音比外面的更大?",
    },
    {
        "condition": lambda results: _has(results, Dimension.EMOTION, "determination") and _has(results, Dimension.EQ, "high_distress"),
        "label": "在硬撑和真强之间?",
    },
]

# === Banned Terms ===
BANNED_TERMS = {
    "抑郁", "焦虑症", "PTSD", "创伤后", "人格障碍", "精神分裂",
    "懦弱", "逃避", "依赖", "自恋", "反社会", "被动攻击",
    "depression", "anxiety disorder", "PTSD", "narcissist", "psychopath",
}


def _has(results: dict[Dimension, dict], dim: Dimension, signal_key: str) -> bool:
    """Check if a dimension has a specific signal activated."""
    if dim not in results:
        return False
    r = results[dim]
    detail = r.get("detail", {})
    # Check various detail formats
    if dim == Dimension.EMOTION:
        top = detail.get("top_emotions", [])
        return any(name == signal_key for name, _ in top[:3])
    elif dim == Dimension.CONFLICT:
        styles = detail.get("styles", {})
        top = max(styles, key=styles.get, default=None) if styles else None
        return top == signal_key
    elif dim == Dimension.FRAGILITY:
        return detail.get("pattern") == signal_key
    elif dim == Dimension.HUMOR:
        styles = detail.get("styles", {})
        top = max(styles, key=styles.get, default=None) if styles else None
        return top == signal_key
    elif dim == Dimension.EQ:
        features = detail.get("features", {})
        if signal_key == "high_self_ref":
            return features.get("self_ref", 0) >= 0.08
        elif signal_key == "high_question_ratio":
            return features.get("question_ratio", 0) >= 0.15
        elif signal_key == "high_distress":
            return detail.get("distress", 0) >= 0.3
        elif signal_key == "negative_valence":
            return detail.get("valence", 0) < -0.2
    elif dim == Dimension.MBTI:
        return signal_key in detail.get("traits", []) or signal_key in detail.get("dimensions", {})
    elif dim == Dimension.LOVE_LANGUAGE:
        return detail.get("primary_language") == signal_key
    return False


def get_positive_label(dim: Dimension, signal_key: str) -> str | None:
    """Get the 70% positive label for a dimension+signal."""
    labels = POSITIVE_LABELS.get(dim, {})
    return labels.get(signal_key)


def get_dark_labels(results: dict[Dimension, dict]) -> list[str]:
    """Get applicable 30% dark star labels from cross-dimensional patterns."""
    found = []
    for pattern in DARK_LABELS:
        try:
            if pattern["condition"](results):
                found.append(pattern["label"])
        except Exception:
            continue
    return found


def get_signal_key(dim: Dimension, detail: dict) -> str | None:
    """Extract the primary signal key from a detector result's detail."""
    if dim == Dimension.EMOTION:
        top = detail.get("top_emotions", [])
        return top[0][0] if top else None
    elif dim == Dimension.CONFLICT:
        styles = detail.get("styles", {})
        return max(styles, key=styles.get, default=None) if styles else None
    elif dim == Dimension.FRAGILITY:
        return detail.get("pattern")
    elif dim == Dimension.HUMOR:
        styles = detail.get("styles", {})
        return max(styles, key=styles.get, default=None) if styles else None
    elif dim == Dimension.EQ:
        features = detail.get("features", {})
        sr = features.get("self_ref", 0)
        qr = features.get("question_ratio", 0)
        distress = detail.get("distress", 0)
        valence = detail.get("valence", 0)
        # Compound signals first (more specific)
        if sr >= 0.08 and qr >= 0.15:
            return "high_self_ref+question"
        if sr >= 0.08 and distress >= 0.3:
            return "high_self_ref+distress"
        # Single signals
        if qr >= 0.15:
            return "high_question_ratio"
        if distress >= 0.3:
            return "high_distress"
        if sr >= 0.08:
            return "high_self_ref"
        if valence < -0.2:
            return "negative_valence"
        if sr < 0.03 and features.get("words", 0) >= 20:
            return "low_self_ref"
    elif dim == Dimension.SOULGRAPH:
        items = detail.get("items", 0)
        spec = detail.get("avg_specificity", 0)
        if items >= 2 and spec >= 0.5:
            return "clear_intention"
        return "exploring"
    return None
