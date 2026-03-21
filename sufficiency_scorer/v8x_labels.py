"""V8x Star Label Mapping — 30/70 Rule.

Maps detector results to human-friendly star labels.
70% bright stars (strengths), 30% dim stars (questions to explore).
Covers all 11 detectors in the SoulMap pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class StarLabel:
    """A resolved star label for display."""
    label: str
    sublabel: str
    star_type: str  # "bright" or "dim"
    star_color: str
    is_question: bool = False


# ── Label rules for all 11 detectors ──────────────────────────────────

LABEL_MAP: dict[str, list[dict]] = {
    "emotion": [
        # bright (70%)
        {"condition": lambda r: r.get("frustration", 0) > 0.5,
         "star_type": "bright", "zh": "对不公平很敏感", "en": "Sensitive to injustice",
         "sublabel": "emotion", "color": "rose"},
        {"condition": lambda r: r.get("sadness", 0) > 0.5,
         "star_type": "bright", "zh": "内心柔软", "en": "Soft-hearted",
         "sublabel": "emotion", "color": "rose"},
        {"condition": lambda r: r.get("happiness", 0) > 0.5,
         "star_type": "bright", "zh": "内心有光", "en": "Inner light",
         "sublabel": "emotion", "color": "rose"},
        {"condition": lambda r: r.get("anger", 0) > 0.5,
         "star_type": "bright", "zh": "有自己的底线", "en": "Holds firm boundaries",
         "sublabel": "emotion", "color": "rose"},
        # dim (30%)
        {"condition": lambda r: r.get("anxiety", 0) > 0.3,
         "star_type": "dim", "zh": "内心有未说出的担忧?", "en": "Unspoken worries inside?",
         "sublabel": "emotion", "color": "rose", "is_question": True},
        {"condition": lambda r: r.get("loneliness", 0) > 0.3,
         "star_type": "dim", "zh": "渴望被真正理解?", "en": "Longing to be truly understood?",
         "sublabel": "emotion", "color": "rose", "is_question": True},
    ],
    "conflict": [
        {"condition": lambda r: r.get("style") == "avoid",
         "star_type": "bright", "zh": "先思考再行动", "en": "Think before you act",
         "sublabel": "conflict", "color": "amber"},
        {"condition": lambda r: r.get("style") == "confront",
         "star_type": "bright", "zh": "直面问题的勇气", "en": "Courage to face problems head-on",
         "sublabel": "conflict", "color": "amber"},
        {"condition": lambda r: r.get("style") == "compromise",
         "star_type": "bright", "zh": "天生的调解者", "en": "Natural mediator",
         "sublabel": "conflict", "color": "amber"},
        {"condition": lambda r: r.get("style") == "collaborate",
         "star_type": "bright", "zh": "寻找双赢的人", "en": "Seeks win-win solutions",
         "sublabel": "conflict", "color": "amber"},
        {"condition": lambda r: r.get("style") == "compete",
         "star_type": "bright", "zh": "战士精神", "en": "Warrior spirit",
         "sublabel": "conflict", "color": "amber"},
    ],
    "fragility": [
        {"condition": lambda r: r.get("pattern") == "open",
         "star_type": "bright", "zh": "敢于面对真实", "en": "Dares to face reality",
         "sublabel": "fragility", "color": "pink"},
        {"condition": lambda r: r.get("pattern") == "defensive",
         "star_type": "bright", "zh": "用坚强保护柔软", "en": "Strength shields softness",
         "sublabel": "fragility", "color": "pink"},
        {"condition": lambda r: r.get("pattern") == "masked",
         "star_type": "bright", "zh": "选择性敞开", "en": "Selectively open",
         "sublabel": "fragility", "color": "pink"},
        {"condition": lambda r: r.get("pattern") == "denial",
         "star_type": "dim", "zh": "隐藏的温柔?", "en": "Hidden tenderness?",
         "sublabel": "fragility", "color": "pink", "is_question": True},
    ],
    "humor": [
        {"condition": lambda r: r.get("style") == "affiliative",
         "star_type": "bright", "zh": "温暖的段子手", "en": "Warm storyteller",
         "sublabel": "humor", "color": "yellow"},
        {"condition": lambda r: r.get("style") == "self_enhancing",
         "star_type": "bright", "zh": "用幽默面对困难", "en": "Humor in the face of hardship",
         "sublabel": "humor", "color": "yellow"},
        {"condition": lambda r: r.get("style") == "aggressive",
         "star_type": "dim", "zh": "毒舌背后是温柔?", "en": "Tenderness behind the sharp tongue?",
         "sublabel": "humor", "color": "yellow", "is_question": True},
        {"condition": lambda r: r.get("style") == "self_deprecating",
         "star_type": "bright", "zh": "自嘲是你的超能力", "en": "Self-deprecation is your superpower",
         "sublabel": "humor", "color": "yellow"},
    ],
    "mbti": [
        {"condition": lambda r: r.get("I", 0) > 0.6,
         "star_type": "bright", "zh": "内心世界丰富", "en": "Rich inner world",
         "sublabel": "mbti", "color": "sky"},
        {"condition": lambda r: r.get("E", 0) > 0.6,
         "star_type": "bright", "zh": "从人群中获取能量", "en": "Energized by people",
         "sublabel": "mbti", "color": "sky"},
        {"condition": lambda r: r.get("N", 0) > 0.6,
         "star_type": "bright", "zh": "直觉敏锐", "en": "Keen intuition",
         "sublabel": "mbti", "color": "sky"},
        {"condition": lambda r: r.get("F", 0) > 0.6,
         "star_type": "bright", "zh": "感受型决策者", "en": "Feeling-driven decision maker",
         "sublabel": "mbti", "color": "sky"},
        {"condition": lambda r: r.get("P", 0) > 0.6,
         "star_type": "bright", "zh": "灵活应变", "en": "Flexible and adaptive",
         "sublabel": "mbti", "color": "sky"},
        {"condition": lambda r: r.get("J", 0) > 0.6,
         "star_type": "bright", "zh": "有条不紊", "en": "Well-organized",
         "sublabel": "mbti", "color": "sky"},
    ],
    "love_language": [
        {"condition": lambda r: r.get("primary") == "words",
         "star_type": "bright", "zh": "渴望被肯定", "en": "Craves affirmation",
         "sublabel": "love_language", "color": "purple"},
        {"condition": lambda r: r.get("primary") == "service",
         "star_type": "bright", "zh": "行动胜过千言", "en": "Actions speak louder",
         "sublabel": "love_language", "color": "purple"},
        {"condition": lambda r: r.get("primary") == "gifts",
         "star_type": "bright", "zh": "用心的礼物最动人", "en": "Heartfelt gifts mean the most",
         "sublabel": "love_language", "color": "purple"},
        {"condition": lambda r: r.get("primary") == "time",
         "star_type": "bright", "zh": "珍惜在一起的时光", "en": "Treasures time together",
         "sublabel": "love_language", "color": "purple"},
        {"condition": lambda r: r.get("primary") == "touch",
         "star_type": "bright", "zh": "温暖的拥抱者", "en": "Warm embracer",
         "sublabel": "love_language", "color": "purple"},
    ],
    "eq": [
        {"condition": lambda r: r.get("self_awareness", 0) > 0.7,
         "star_type": "bright", "zh": "情绪觉察者", "en": "Emotionally self-aware",
         "sublabel": "eq", "color": "green"},
        {"condition": lambda r: r.get("self_regulation", 0) > 0.7,
         "star_type": "bright", "zh": "善于调节自己", "en": "Skilled at self-regulation",
         "sublabel": "eq", "color": "green"},
        {"condition": lambda r: r.get("empathy", 0) > 0.7,
         "star_type": "bright", "zh": "共情能力很强", "en": "Highly empathetic",
         "sublabel": "eq", "color": "green"},
        {"condition": lambda r: r.get("social_skills", 0) > 0.7,
         "star_type": "bright", "zh": "读懂弦外之音", "en": "Reads between the lines",
         "sublabel": "eq", "color": "green"},
    ],
    "attachment": [
        {"condition": lambda r: r.get("style") == "secure",
         "star_type": "bright", "zh": "安全感充足", "en": "Securely attached",
         "sublabel": "attachment", "color": "lavender"},
        {"condition": lambda r: r.get("style") == "anxious",
         "star_type": "dim", "zh": "害怕失去?", "en": "Afraid of losing?",
         "sublabel": "attachment", "color": "lavender", "is_question": True},
        {"condition": lambda r: r.get("style") == "avoidant",
         "star_type": "bright", "zh": "独立是你的铠甲", "en": "Independence is your armor",
         "sublabel": "attachment", "color": "lavender"},
        {"condition": lambda r: r.get("style") == "fearful",
         "star_type": "dim", "zh": "想靠近又怕受伤?", "en": "Want to get close but afraid of getting hurt?",
         "sublabel": "attachment", "color": "lavender", "is_question": True},
    ],
    "values": [
        {"condition": lambda r: r.get("primary") == "self_direction",
         "star_type": "bright", "zh": "自由是第一需求", "en": "Freedom comes first",
         "sublabel": "values", "color": "gold"},
        {"condition": lambda r: r.get("primary") == "benevolence",
         "star_type": "bright", "zh": "善良是你的底色", "en": "Kindness is your foundation",
         "sublabel": "values", "color": "gold"},
        {"condition": lambda r: r.get("primary") == "achievement",
         "star_type": "bright", "zh": "追求卓越", "en": "Pursuit of excellence",
         "sublabel": "values", "color": "gold"},
        {"condition": lambda r: r.get("primary") == "security",
         "star_type": "bright", "zh": "稳定是力量", "en": "Stability is strength",
         "sublabel": "values", "color": "gold"},
    ],
    "behavioral": [
        {"condition": lambda r: r.get("question_ratio", 0) > 0.2,
         "star_type": "bright", "zh": "主动寻找出路", "en": "Actively seeks solutions",
         "sublabel": "behavioral", "color": "teal"},
        {"condition": lambda r: r.get("self_ref_ratio", 0) > 0.2,
         "star_type": "bright", "zh": "善于自我觉察", "en": "Self-aware",
         "sublabel": "behavioral", "color": "teal"},
        {"condition": lambda r: r.get("hedging_ratio", 0) > 0.1,
         "star_type": "dim", "zh": "内心还在犹豫?", "en": "Still hesitating inside?",
         "sublabel": "behavioral", "color": "teal", "is_question": True},
    ],
    "connection_response": [
        {"condition": lambda r: r.get("style") == "connected",
         "star_type": "bright", "zh": "善于建立连接", "en": "Skilled at building connections",
         "sublabel": "connection_response", "color": "teal"},
        {"condition": lambda r: r.get("style") == "selective",
         "star_type": "bright", "zh": "选择性敞开", "en": "Selectively open",
         "sublabel": "connection_response", "color": "teal"},
        {"condition": lambda r: r.get("style") == "detached",
         "star_type": "dim", "zh": "保持距离是一种保护?", "en": "Keeping distance as protection?",
         "sublabel": "connection_response", "color": "teal", "is_question": True},
    ],
}


def get_star_label(
    detector: str,
    results: dict,
    lang: str = "zh",
    force_dim: bool = False,
) -> StarLabel | None:
    """Find the first matching star label for a detector's results.

    Args:
        detector: detector name (e.g. "emotion", "conflict")
        results: detector output dict
        lang: "zh" or "en"
        force_dim: if True, prefer dim star labels
    Returns:
        StarLabel or None if no match / unknown detector
    """
    rules = LABEL_MAP.get(detector)
    if rules is None:
        return None

    if force_dim:
        # Try dim rules first
        for rule in rules:
            if rule["star_type"] == "dim":
                try:
                    if rule["condition"](results):
                        return _rule_to_label(rule, lang)
                except Exception:
                    continue
        # Fall back to any match
        for rule in rules:
            try:
                if rule["condition"](results):
                    return _rule_to_label(rule, lang)
            except Exception:
                continue
        return None

    # Normal: first match wins
    for rule in rules:
        try:
            if rule["condition"](results):
                return _rule_to_label(rule, lang)
        except Exception:
            continue
    return None


def _rule_to_label(rule: dict, lang: str) -> StarLabel:
    """Convert a rule dict to a StarLabel."""
    return StarLabel(
        label=rule[lang],
        sublabel=rule["sublabel"],
        star_type=rule["star_type"],
        star_color=rule["color"],
        is_question=rule.get("is_question", False),
    )
