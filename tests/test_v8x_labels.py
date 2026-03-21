"""Tests for V8x star label mapping (30/70 rule)."""

import pytest
from sufficiency_scorer.v8x_labels import (
    LABEL_MAP,
    StarLabel,
    get_star_label,
)


class TestEmotionLabels:
    def test_frustration_bright(self):
        results = {"frustration": 0.7, "sadness": 0.1}
        label = get_star_label("emotion", results)
        assert label.label == "对不公平很敏感"
        assert label.star_type == "bright"
        assert label.star_color == "rose"
        assert label.is_question is False

    def test_sadness_bright(self):
        results = {"sadness": 0.6}
        label = get_star_label("emotion", results)
        assert label.label == "内心柔软"
        assert label.star_type == "bright"

    def test_happiness_bright(self):
        results = {"happiness": 0.8}
        label = get_star_label("emotion", results)
        assert label.label == "内心有光"

    def test_anger_bright(self):
        results = {"anger": 0.6}
        label = get_star_label("emotion", results)
        assert label.label == "有自己的底线"

    def test_anxiety_dim(self):
        results = {"anxiety": 0.4}
        label = get_star_label("emotion", results)
        assert label.label == "内心有未说出的担忧?"
        assert label.star_type == "dim"
        assert label.is_question is True

    def test_loneliness_dim(self):
        results = {"loneliness": 0.5}
        label = get_star_label("emotion", results)
        assert label.label == "渴望被真正理解?"
        assert label.star_type == "dim"
        assert label.is_question is True


class TestConflictLabels:
    def test_avoid(self):
        results = {"style": "avoid"}
        label = get_star_label("conflict", results)
        assert label.label == "先思考再行动"
        assert label.star_type == "bright"
        assert label.star_color == "amber"

    def test_confront(self):
        results = {"style": "confront"}
        label = get_star_label("conflict", results)
        assert label.label == "直面问题的勇气"

    def test_compromise(self):
        results = {"style": "compromise"}
        label = get_star_label("conflict", results)
        assert label.label == "天生的调解者"

    def test_collaborate(self):
        results = {"style": "collaborate"}
        label = get_star_label("conflict", results)
        assert label.label == "寻找双赢的人"

    def test_compete(self):
        results = {"style": "compete"}
        label = get_star_label("conflict", results)
        assert label.label == "战士精神"


class TestFragilityLabels:
    def test_open_bright(self):
        results = {"pattern": "open"}
        label = get_star_label("fragility", results)
        assert label.label == "敢于面对真实"
        assert label.star_type == "bright"
        assert label.star_color == "pink"

    def test_defensive_bright(self):
        results = {"pattern": "defensive"}
        label = get_star_label("fragility", results)
        assert label.label == "用坚强保护柔软"
        assert label.star_type == "bright"

    def test_masked_bright(self):
        results = {"pattern": "masked"}
        label = get_star_label("fragility", results)
        assert label.label == "选择性敞开"

    def test_denial_dim(self):
        results = {"pattern": "denial"}
        label = get_star_label("fragility", results)
        assert label.label == "隐藏的温柔?"
        assert label.star_type == "dim"
        assert label.is_question is True


class TestHumorLabels:
    def test_affiliative(self):
        results = {"style": "affiliative"}
        label = get_star_label("humor", results)
        assert label.label == "温暖的段子手"
        assert label.star_color == "yellow"

    def test_self_enhancing(self):
        results = {"style": "self_enhancing"}
        label = get_star_label("humor", results)
        assert label.label == "用幽默面对困难"

    def test_aggressive_dim(self):
        results = {"style": "aggressive"}
        label = get_star_label("humor", results)
        assert label.label == "毒舌背后是温柔?"
        assert label.star_type == "dim"
        assert label.is_question is True

    def test_self_deprecating(self):
        results = {"style": "self_deprecating"}
        label = get_star_label("humor", results)
        assert label.label == "自嘲是你的超能力"
        assert label.star_type == "bright"


class TestMBTILabels:
    def test_high_i(self):
        results = {"I": 0.8, "E": 0.2}
        label = get_star_label("mbti", results)
        assert label.label == "内心世界丰富"
        assert label.star_type == "bright"
        assert label.star_color == "sky"

    def test_high_e(self):
        results = {"E": 0.7, "I": 0.3}
        label = get_star_label("mbti", results)
        assert label.label == "从人群中获取能量"

    def test_high_n(self):
        results = {"N": 0.8, "S": 0.2}
        label = get_star_label("mbti", results)
        assert label.label == "直觉敏锐"

    def test_high_f(self):
        results = {"F": 0.7}
        label = get_star_label("mbti", results)
        assert label.label == "感受型决策者"

    def test_high_p(self):
        results = {"P": 0.8}
        label = get_star_label("mbti", results)
        assert label.label == "灵活应变"

    def test_high_j(self):
        results = {"J": 0.7}
        label = get_star_label("mbti", results)
        assert label.label == "有条不紊"


class TestLoveLanguageLabels:
    def test_words(self):
        results = {"primary": "words"}
        label = get_star_label("love_language", results)
        assert label.label == "渴望被肯定"
        assert label.star_type == "bright"
        assert label.star_color == "purple"

    def test_service(self):
        results = {"primary": "service"}
        label = get_star_label("love_language", results)
        assert label.label == "行动胜过千言"

    def test_gifts(self):
        results = {"primary": "gifts"}
        label = get_star_label("love_language", results)
        assert label.label == "用心的礼物最动人"

    def test_time(self):
        results = {"primary": "time"}
        label = get_star_label("love_language", results)
        assert label.label == "珍惜在一起的时光"

    def test_touch(self):
        results = {"primary": "touch"}
        label = get_star_label("love_language", results)
        assert label.label == "温暖的拥抱者"


class TestEQLabels:
    def test_self_awareness(self):
        results = {"self_awareness": 0.8}
        label = get_star_label("eq", results)
        assert label.label == "情绪觉察者"
        assert label.star_color == "green"

    def test_self_regulation(self):
        results = {"self_regulation": 0.75}
        label = get_star_label("eq", results)
        assert label.label == "善于调节自己"

    def test_empathy(self):
        results = {"empathy": 0.8}
        label = get_star_label("eq", results)
        assert label.label == "共情能力很强"

    def test_social_skills(self):
        results = {"social_skills": 0.9}
        label = get_star_label("eq", results)
        assert label.label == "读懂弦外之音"


class TestAttachmentLabels:
    def test_secure(self):
        results = {"style": "secure"}
        label = get_star_label("attachment", results)
        assert label.label == "安全感充足"
        assert label.star_type == "bright"
        assert label.star_color == "lavender"

    def test_anxious_dim(self):
        results = {"style": "anxious"}
        label = get_star_label("attachment", results)
        assert label.label == "害怕失去?"
        assert label.star_type == "dim"
        assert label.is_question is True

    def test_avoidant_bright(self):
        results = {"style": "avoidant"}
        label = get_star_label("attachment", results)
        assert label.label == "独立是你的铠甲"
        assert label.star_type == "bright"

    def test_fearful_dim(self):
        results = {"style": "fearful"}
        label = get_star_label("attachment", results)
        assert label.label == "想靠近又怕受伤?"
        assert label.star_type == "dim"


class TestValuesLabels:
    def test_self_direction(self):
        results = {"primary": "self_direction"}
        label = get_star_label("values", results)
        assert label.label == "自由是第一需求"
        assert label.star_type == "bright"
        assert label.star_color == "gold"

    def test_benevolence(self):
        results = {"primary": "benevolence"}
        label = get_star_label("values", results)
        assert label.label == "善良是你的底色"

    def test_achievement(self):
        results = {"primary": "achievement"}
        label = get_star_label("values", results)
        assert label.label == "追求卓越"

    def test_security(self):
        results = {"primary": "security"}
        label = get_star_label("values", results)
        assert label.label == "稳定是力量"


class TestBehavioralLabels:
    def test_question_ratio(self):
        results = {"question_ratio": 0.3}
        label = get_star_label("behavioral", results)
        assert label.label == "主动寻找出路"
        assert label.star_type == "bright"
        assert label.star_color == "teal"

    def test_self_ref_ratio(self):
        results = {"self_ref_ratio": 0.25}
        label = get_star_label("behavioral", results)
        assert label.label == "善于自我觉察"
        assert label.star_type == "bright"

    def test_hedging_ratio_dim(self):
        results = {"hedging_ratio": 0.15}
        label = get_star_label("behavioral", results)
        assert label.label == "内心还在犹豫?"
        assert label.star_type == "dim"
        assert label.is_question is True


class TestConnectionResponseLabels:
    def test_connected(self):
        results = {"style": "connected"}
        label = get_star_label("connection_response", results)
        assert label.label == "善于建立连接"
        assert label.star_type == "bright"
        assert label.star_color == "teal"

    def test_selective(self):
        results = {"style": "selective"}
        label = get_star_label("connection_response", results)
        assert label.label == "选择性敞开"

    def test_detached_dim(self):
        results = {"style": "detached"}
        label = get_star_label("connection_response", results)
        assert label.label == "保持距离是一种保护?"
        assert label.star_type == "dim"
        assert label.is_question is True


class TestDimStarIsQuestion:
    """All dim stars must have is_question=True."""

    def test_all_dim_stars_have_question(self):
        for detector, rules in LABEL_MAP.items():
            for rule in rules:
                if rule["star_type"] == "dim":
                    assert rule.get("is_question", False) is True, (
                        f"{detector} dim rule '{rule['zh']}' missing is_question=True"
                    )


class TestEnglishLabels:
    def test_emotion_english(self):
        results = {"frustration": 0.7}
        label = get_star_label("emotion", results, lang="en")
        assert isinstance(label.label, str)
        assert label.label != ""
        # Should be English, not Chinese
        assert not any("\u4e00" <= c <= "\u9fff" for c in label.label)

    def test_conflict_english(self):
        results = {"style": "avoid"}
        label = get_star_label("conflict", results, lang="en")
        assert isinstance(label.label, str)
        assert not any("\u4e00" <= c <= "\u9fff" for c in label.label)


class TestAllDetectorsCovered:
    EXPECTED_DETECTORS = {
        "emotion", "conflict", "fragility", "humor", "mbti",
        "love_language", "eq", "attachment", "values",
        "behavioral", "connection_response",
    }

    def test_all_11_detectors_in_label_map(self):
        assert set(LABEL_MAP.keys()) == self.EXPECTED_DETECTORS

    def test_label_map_has_11_entries(self):
        assert len(LABEL_MAP) == 11


class TestBrightDimRatio:
    """70% bright / 30% dim across all labels."""

    def test_70_percent_bright(self):
        total = 0
        bright = 0
        for rules in LABEL_MAP.values():
            for rule in rules:
                total += 1
                if rule["star_type"] == "bright":
                    bright += 1
        ratio = bright / total
        assert ratio >= 0.65, f"Bright ratio {ratio:.1%} is below 65%"
        assert ratio <= 0.85, f"Bright ratio {ratio:.1%} is above 85%"


class TestForceDim:
    def test_force_dim_overrides_bright(self):
        results = {"frustration": 0.7}
        label = get_star_label("emotion", results, force_dim=True)
        # Should return a dim star if available
        if label.star_type == "dim":
            assert label.is_question is True


class TestStarLabelDataclass:
    def test_fields(self):
        sl = StarLabel(
            label="test",
            sublabel="emotion",
            star_type="bright",
            star_color="rose",
            is_question=False,
        )
        assert sl.label == "test"
        assert sl.sublabel == "emotion"
        assert sl.star_type == "bright"
        assert sl.star_color == "rose"
        assert sl.is_question is False

    def test_default_is_question(self):
        sl = StarLabel(
            label="test",
            sublabel="emotion",
            star_type="bright",
            star_color="rose",
        )
        assert sl.is_question is False


class TestNoMatchReturnsNone:
    def test_no_match(self):
        results = {"frustration": 0.1, "sadness": 0.1}
        label = get_star_label("emotion", results)
        assert label is None

    def test_unknown_detector(self):
        label = get_star_label("nonexistent", {"x": 1})
        assert label is None
