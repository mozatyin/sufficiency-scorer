"""Insight extraction — the AlphaGo layer.

Two passes:
1. Cross-dimensional (HIGH): surprising patterns across detector pairs.
2. Single-dimensional (MEDIUM): top signal from each detector, positively reframed.
"""

import re

from sufficiency_scorer.config import MIN_CONFIDENCE, MIN_INSIGHT_CONFIDENCE
from sufficiency_scorer.models import (
    DetectorResult,
    Dimension,
    InsightCandidate,
    InsightQuality,
)

# ---------------------------------------------------------------------------
# Single-dimension reframe maps
# ---------------------------------------------------------------------------

EMOTION_REFRAMES: dict[str, str] = {
    "frustration": "You have a sharp sense of when things aren't right",
    "anger": "You care deeply about fairness",
    "sadness": "You feel things at a depth most people don't reach",
    "fear": "You're tuned into risks others might miss",
    "anxiety": "Your mind is always working to protect you",
    "confusion": "You're honest enough to sit with uncertainty instead of faking clarity",
    "hope": "Even in difficulty, you hold onto what matters",
    "despair": "You've been carrying something heavy for a while",
    "grief": "The depth of your loss reflects the depth of what you had",
    "irritation": "You notice friction that others overlook",
    "joy": "You let good moments land — that's rarer than it sounds",
    "surprise": "You stay open to the unexpected instead of filtering it out",
    "disgust": "You have strong standards for how things should be",
    "contempt": "You see clearly when something falls short — that clarity has power",
    "love": "You connect with a warmth that others can feel",
    "guilt": "You hold yourself to a high standard — that speaks to your integrity",
    "shame": "You care about being the person you want to be",
    "loneliness": "You value deep connection, not just company",
    "jealousy": "You know what you want — that's clarity, not weakness",
    "pride": "You recognize your own worth — that confidence is earned",
}

CONFLICT_REFRAMES: dict[str, str] = {
    "avoid": "You think before you engage — that's restraint, not weakness",
    "confront": "You face things head-on instead of letting them fester",
    "compromise": "You look for the middle ground — that's practical wisdom",
    "collaborate": "You instinctively look for the win-win — that's rare",
    "compete": "When it matters, you fight for what you believe in",
    "avoidance": "You think before you engage — that's restraint, not weakness",
    "confrontation": "You face things head-on instead of letting them fester",
    "accommodating": "You put others first — that generosity runs deep",
}

HUMOR_REFRAMES: dict[str, str] = {
    "self_deprecating": "You use humor to keep things real — people trust that",
    "affiliative": "Your humor brings people together — that's a social superpower",
    "aggressive": "Your wit cuts through pretense — you don't sugarcoat",
    "self_enhancing": "You find the funny side even in hard moments — that's resilience",
}

FRAGILITY_REFRAMES: dict[str, str] = {
    "open": "You let yourself be seen — that takes more courage than most realize",
    "defensive": "You've built ways to protect yourself — and that kept you safe when you needed it",
    "masked": "You carry more than you show — there's depth behind the surface",
    "denial": "You've found a way to keep going — and when you're ready, there's more to explore",
}

EQ_REFRAMES: dict[str, str] = {
    "high_question_ratio": "You're not just talking — you're actively searching for understanding",
    "high_distress": "You're feeling the weight of something real right now",
    "negative_valence": "You're going through a hard stretch — acknowledging that is the first step",
    "high_self_ref": "You're deeply self-aware — you pay attention to your own inner world",
}


# ---------------------------------------------------------------------------
# Cross-dimensional pattern definitions
# ---------------------------------------------------------------------------

class _CrossPattern:
    """A pattern that matches across two detector dimensions."""

    __slots__ = ("dim_a", "dim_b", "match_fn", "signal_fn", "reframe")

    def __init__(self, dim_a, dim_b, match_fn, signal_fn, reframe):
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.match_fn = match_fn  # (detail_a, detail_b) -> bool
        self.signal_fn = signal_fn  # (detail_a, detail_b) -> str
        self.reframe = reframe


def _top_emotion(detail: dict) -> tuple[str, float] | None:
    """Return (name, score) of top emotion, or None."""
    tops = detail.get("top_emotions", [])
    if tops and len(tops) > 0:
        return tops[0]
    return None


def _has_emotion(detail: dict, name: str) -> bool:
    te = _top_emotion(detail)
    return te is not None and te[0] == name


def _any_strong_emotion(detail: dict, threshold: float = 0.3) -> bool:
    te = _top_emotion(detail)
    return te is not None and te[1] >= threshold


def _top_conflict(detail: dict) -> tuple[str, float] | None:
    styles = detail.get("styles", {})
    if not styles:
        return None
    best = max(styles.items(), key=lambda x: x[1])
    return best


def _has_conflict(detail: dict, name: str) -> bool:
    tc = _top_conflict(detail)
    return tc is not None and tc[0] == name


def _top_humor(detail: dict) -> tuple[str, float] | None:
    styles = detail.get("styles", {})
    if not styles:
        return None
    best = max(styles.items(), key=lambda x: x[1])
    return best


def _has_humor(detail: dict, name: str) -> bool:
    th = _top_humor(detail)
    return th is not None and th[0] == name


def _fragility_pattern(detail: dict) -> str | None:
    return detail.get("pattern")


CROSS_PATTERNS: list[_CrossPattern] = [
    # 1. frustration + avoidance
    _CrossPattern(
        Dimension.EMOTION, Dimension.CONFLICT,
        lambda e, c: _has_emotion(e, "frustration") and _has_conflict(c, "avoid"),
        lambda e, c: f"frustration ({_top_emotion(e)[1]:.0%}) + avoidance ({_top_conflict(c)[1]:.0%})",
        "You feel the friction but choose restraint — you're more strategic than you give yourself credit for",
    ),
    # 2. anger + confrontation
    _CrossPattern(
        Dimension.EMOTION, Dimension.CONFLICT,
        lambda e, c: _has_emotion(e, "anger") and _has_conflict(c, "confront"),
        lambda e, c: f"anger ({_top_emotion(e)[1]:.0%}) + confrontation ({_top_conflict(c)[1]:.0%})",
        "When you see injustice, you don't look away — that takes backbone",
    ),
    # 3. self-deprecating humor + open fragility
    _CrossPattern(
        Dimension.HUMOR, Dimension.FRAGILITY,
        lambda h, f: _has_humor(h, "self_deprecating") and _fragility_pattern(f) == "open",
        lambda h, f: f"self-deprecating humor ({_top_humor(h)[1]:.0%}) + open fragility",
        "You laugh at yourself not to hide, but to stay real — humor is your honesty tool",
    ),
    # 4. self-deprecating humor + masked fragility
    _CrossPattern(
        Dimension.HUMOR, Dimension.FRAGILITY,
        lambda h, f: _has_humor(h, "self_deprecating") and _fragility_pattern(f) == "masked",
        lambda h, f: f"self-deprecating humor ({_top_humor(h)[1]:.0%}) + masked fragility",
        "Your jokes carry more weight than people realize — there's something underneath worth exploring",
    ),
    # 5. any strong emotion + high self_ref
    _CrossPattern(
        Dimension.EMOTION, Dimension.EQ,
        lambda e, q: _any_strong_emotion(e) and q.get("features", {}).get("self_ref", 0) >= 0.1,
        lambda e, q: f"{_top_emotion(e)[0]} ({_top_emotion(e)[1]:.0%}) + self-awareness (self_ref {q.get('features', {}).get('self_ref', 0):.0%})",
        "You're deeply self-aware — you don't just feel things, you try to understand them",
    ),
    # 6. any strong emotion + high question_ratio
    _CrossPattern(
        Dimension.EMOTION, Dimension.EQ,
        lambda e, q: _any_strong_emotion(e) and q.get("features", {}).get("question_ratio", 0) >= 0.2,
        lambda e, q: f"{_top_emotion(e)[0]} ({_top_emotion(e)[1]:.0%}) + questioning ({q.get('features', {}).get('question_ratio', 0):.0%})",
        "You're not just venting — you're actively searching for answers",
    ),
    # 7. sadness + open fragility
    _CrossPattern(
        Dimension.EMOTION, Dimension.FRAGILITY,
        lambda e, f: _has_emotion(e, "sadness") and _fragility_pattern(f) == "open",
        lambda e, f: f"sadness ({_top_emotion(e)[1]:.0%}) + open fragility",
        "You let yourself feel the hard things instead of running — that's braver than most people manage",
    ),
    # 8. anxiety + defensive fragility
    _CrossPattern(
        Dimension.EMOTION, Dimension.FRAGILITY,
        lambda e, f: _has_emotion(e, "anxiety") and _fragility_pattern(f) == "defensive",
        lambda e, f: f"anxiety ({_top_emotion(e)[1]:.0%}) + defensive fragility",
        "You've built a shield, but the fact that you're here means part of you is ready to set it down",
    ),
    # 9. avoidance + high self_ref
    _CrossPattern(
        Dimension.CONFLICT, Dimension.EQ,
        lambda c, q: _has_conflict(c, "avoid") and q.get("features", {}).get("self_ref", 0) >= 0.1,
        lambda c, q: f"avoidance ({_top_conflict(c)[1]:.0%}) + self-awareness (self_ref {q.get('features', {}).get('self_ref', 0):.0%})",
        "You think before you act in conflict — that's emotional intelligence, not avoidance",
    ),
    # 10. open fragility + negative valence
    _CrossPattern(
        Dimension.FRAGILITY, Dimension.EQ,
        lambda f, q: _fragility_pattern(f) == "open" and q.get("valence", 0) < -0.2,
        lambda f, q: f"open fragility + negative valence ({q.get('valence', 0):.2f})",
        "You're going through something hard and you're not pretending otherwise — that honesty is your strength",
    ),
    # 11. anger + avoidance (suppressed anger)
    _CrossPattern(
        Dimension.EMOTION, Dimension.CONFLICT,
        lambda e, c: _has_emotion(e, "anger") and _has_conflict(c, "avoid"),
        lambda e, c: f"anger ({_top_emotion(e)[1]:.0%}) + avoidance ({_top_conflict(c)[1]:.0%})",
        "You feel strongly but hold back — you're managing more than people see",
    ),
    # 12. high distress + open fragility
    _CrossPattern(
        Dimension.EQ, Dimension.FRAGILITY,
        lambda q, f: q.get("distress", 0) >= 0.3 and _fragility_pattern(f) == "open",
        lambda q, f: f"distress ({q.get('distress', 0):.0%}) + open fragility",
        "You're not hiding from what's hard — that openness is the foundation for growth",
    ),
]


# ---------------------------------------------------------------------------
# InsightExtractor
# ---------------------------------------------------------------------------

class InsightExtractor:
    """Extracts reframeable insights from detector results."""

    def extract(self, results: list[DetectorResult], user_text: str = "") -> list[InsightCandidate]:
        """Run both passes and return sorted insights."""
        active = [r for r in results if r.activated and r.confidence >= MIN_CONFIDENCE]
        by_dim: dict[Dimension, DetectorResult] = {r.dimension: r for r in active}

        insights: list[InsightCandidate] = []
        used_dims: set[Dimension] = set()  # track dimensions consumed by cross-insights

        # --- Pass 1: cross-dimensional (HIGH) ---
        cross_insights = self._cross_dimensional(by_dim)
        for ci in cross_insights:
            insights.append(ci)
            used_dims.update(ci.source_dimensions)

        # --- Pass 2: single-dimensional (MEDIUM) ---
        single_insights = self._single_dimensional(by_dim, used_dims)
        insights.extend(single_insights)

        # --- Contextualize with user text (dynamic reframing) ---
        if user_text:
            context = self._extract_context(user_text)
            insights = [self._contextualize(i, context) for i in insights]

        # Sort: quality desc, confidence desc
        insights.sort(key=lambda i: (-i.quality.value, -i.confidence))
        return insights

    @staticmethod
    def _extract_context(text: str) -> dict:
        """Extract key contextual elements from user text."""
        text_lower = text.lower()
        words = text_lower.split()

        # Extract topic indicators
        topics: list[str] = []
        topic_patterns = {
            "work": ["work", "job", "boss", "overtime", "ot", "office", "career", "colleague"],
            "relationship": ["partner", "husband", "wife", "boyfriend", "girlfriend", "relationship", "marriage", "dating", "dumped", "breakup"],
            "family": ["mom", "dad", "mother", "father", "parent", "brother", "sister", "family", "child", "son", "daughter"],
            "health": ["sleep", "insomnia", "tired", "exhausted", "sick", "anxiety", "panic", "pain"],
            "identity": ["who am i", "purpose", "meaning", "lost", "confused", "direction", "figure out"],
            "grief": ["passed away", "died", "death", "lost", "funeral", "grief", "mourning", "gone"],
        }
        for topic, keywords in topic_patterns.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        # Extract key action phrases (what the user is doing/experiencing)
        actions: list[str] = []
        action_patterns = [
            r"(forcing me to [\w ]{1,30})",
            r"(keeps? (?:pushing|asking|telling|making) me[\w ]{0,20})",
            r"(don'?t know (?:what to do|how to|if I)[\w ]{0,15})",
            r"(can'?t (?:sleep|stop|take|handle|figure)[\w ]{0,20})",
            r"(feel(?:s|ing)? (?:like|trapped|lost|alone|invisible)[\w ]{0,15})",
            r"(got dumped|broke up|left me)",
            r"(passed away|died)",
        ]
        for pattern in action_patterns:
            match = re.search(pattern, text_lower)
            if match:
                actions.append(match.group(1).strip())

        # Extract key nouns for specificity
        specifics: list[str] = []
        for word in words:
            if word in ("boss", "partner", "mom", "dad", "job", "interview", "painting"):
                specifics.append(word)

        return {
            "topics": topics,
            "actions": actions,
            "specifics": specifics,
            "has_question": "?" in text,
        }

    def _contextualize(self, insight: InsightCandidate, context: dict) -> InsightCandidate:
        """Add user context to a reframe if possible."""
        topics = context.get("topics", [])
        actions = context.get("actions", [])
        specifics = context.get("specifics", [])

        if not (topics or actions or specifics):
            return insight

        # Build a contextual suffix
        suffix = ""
        if actions:
            suffix = f" \u2014 especially when {actions[0]}"
        elif specifics and topics:
            topic = topics[0]
            specific = specifics[0]
            topic_phrases = {
                "work": f"at work with your {specific}" if specific in ("boss", "colleague") else "in your work situation",
                "relationship": "in your relationship" if specific in ("partner", "husband", "wife") else "in your relationships",
                "family": f"with your {specific}" if specific in ("mom", "dad", "mother", "father") else "in your family",
                "grief": f"after losing your {specific}" if specific in ("mom", "dad") else "through this loss",
                "health": "with what you're going through physically",
                "identity": "as you figure out what you really want",
            }
            suffix = f" \u2014 {topic_phrases.get(topic, 'in your situation')}"
        elif topics:
            topic_generic = {
                "work": "in your work situation",
                "relationship": "in your relationships",
                "family": "within your family",
                "grief": "through this loss",
                "health": "with what you're going through",
                "identity": "as you search for clarity",
            }
            suffix = f" \u2014 {topic_generic.get(topics[0], 'in your situation')}"

        if suffix and not insight.reframe.endswith(suffix):
            new_reframe = insight.reframe.rstrip(".!") + suffix
            return insight.model_copy(update={"reframe": new_reframe})

        return insight

    # ------------------------------------------------------------------
    # Pass 1 — cross-dimensional
    # ------------------------------------------------------------------

    def _cross_dimensional(self, by_dim: dict[Dimension, DetectorResult]) -> list[InsightCandidate]:
        found: list[InsightCandidate] = []
        # Track which (dim_a, dim_b) pairs have already matched to avoid
        # emitting multiple cross-insights from the same pair
        used_pairs: set[tuple[Dimension, Dimension]] = set()

        for pat in CROSS_PATTERNS:
            if pat.dim_a not in by_dim or pat.dim_b not in by_dim:
                continue
            pair_key = (pat.dim_a, pat.dim_b)
            if pair_key in used_pairs:
                continue

            detail_a = by_dim[pat.dim_a].detail
            detail_b = by_dim[pat.dim_b].detail

            try:
                if not pat.match_fn(detail_a, detail_b):
                    continue
            except (KeyError, TypeError, IndexError):
                continue

            try:
                signal = pat.signal_fn(detail_a, detail_b)
            except (KeyError, TypeError, IndexError):
                signal = f"{pat.dim_a.value} + {pat.dim_b.value}"

            conf = min(by_dim[pat.dim_a].confidence, by_dim[pat.dim_b].confidence)
            if conf < MIN_INSIGHT_CONFIDENCE:
                continue

            found.append(InsightCandidate(
                source_dimensions=[pat.dim_a, pat.dim_b],
                signal=signal,
                reframe=pat.reframe,
                quality=InsightQuality.HIGH,
                confidence=conf,
            ))
            used_pairs.add(pair_key)

        return found

    # ------------------------------------------------------------------
    # Pass 2 — single-dimensional
    # ------------------------------------------------------------------

    def _single_dimensional(
        self,
        by_dim: dict[Dimension, DetectorResult],
        used_dims: set[Dimension],
    ) -> list[InsightCandidate]:
        found: list[InsightCandidate] = []

        for dim, result in by_dim.items():
            # Skip dimensions already consumed by cross-insights
            if dim in used_dims:
                continue

            insight = self._single_for(dim, result)
            if insight is not None:
                found.append(insight)

        return found

    def _single_for(self, dim: Dimension, result: DetectorResult) -> InsightCandidate | None:
        detail = result.detail
        signal: str | None = None
        reframe: str | None = None

        if dim == Dimension.EMOTION:
            te = _top_emotion(detail)
            if te:
                emo_name, emo_score = te
                reframe = EMOTION_REFRAMES.get(emo_name)
                signal = f"top emotion: {emo_name} ({emo_score:.0%})"

        elif dim == Dimension.CONFLICT:
            tc = _top_conflict(detail)
            if tc:
                style, score = tc
                reframe = CONFLICT_REFRAMES.get(style)
                signal = f"conflict style: {style} ({score:.0%})"

        elif dim == Dimension.HUMOR:
            th = _top_humor(detail)
            if th:
                style, score = th
                reframe = HUMOR_REFRAMES.get(style)
                signal = f"humor style: {style} ({score:.0%})"

        elif dim == Dimension.FRAGILITY:
            pat = _fragility_pattern(detail)
            if pat:
                reframe = FRAGILITY_REFRAMES.get(pat)
                signal = f"fragility pattern: {pat}"

        elif dim == Dimension.EQ:
            signal, reframe = self._eq_insight(detail)

        else:
            # Dimensions without specific reframes (MBTI, SOULGRAPH, etc.)
            # produce no single-dimensional insight
            return None

        if signal is None or reframe is None:
            return None

        return InsightCandidate(
            source_dimensions=[dim],
            signal=signal,
            reframe=reframe,
            quality=InsightQuality.MEDIUM,
            confidence=result.confidence,
        )

    @staticmethod
    def _eq_insight(detail: dict) -> tuple[str | None, str | None]:
        features = detail.get("features", {})
        qr = features.get("question_ratio", 0)
        sr = features.get("self_ref", 0)
        distress = detail.get("distress", 0)
        valence = detail.get("valence", 0)

        # Pick the most salient EQ signal
        if qr >= 0.2:
            return f"question_ratio={qr:.0%}", EQ_REFRAMES["high_question_ratio"]
        if distress >= 0.3:
            return f"distress={distress:.2f}", EQ_REFRAMES["high_distress"]
        if valence < -0.2:
            return f"valence={valence:.2f}", EQ_REFRAMES["negative_valence"]
        if sr >= 0.1:
            return f"self_ref={sr:.0%}", EQ_REFRAMES["high_self_ref"]
        return None, None
