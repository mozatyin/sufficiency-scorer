"""Zero-cost precomputation — extract everything possible without LLM.

Runs BEFORE the LLM call. Results are passed as structured signals
so the LLM doesn't waste tokens rediscovering what text analysis can find.
"""

import re

from sufficiency_scorer.detectors.eq import extract_behavioral, compute_valence, compute_distress


def precompute(text: str) -> dict:
    """Extract all zero-cost signals from user text. <1ms."""
    features = extract_behavioral(text)
    valence = compute_valence(features)
    distress = compute_distress(features)
    topics = detect_topics(text)
    tone = detect_tone(text)
    key_phrases = extract_key_phrases(text)
    emotion_words = detect_emotion_words(text)

    return {
        "word_count": features["words"],
        "self_ref": round(features["self_ref"], 3),
        "question_ratio": round(features["question_ratio"], 3),
        "valence": round(valence, 3),
        "distress": round(distress, 3),
        "topics": topics,
        "tone": tone,
        "key_phrases": key_phrases,
        "emotion_words": emotion_words,
    }


def detect_topics(text: str) -> list[str]:
    """Detect topic areas from keywords. Zero cost."""
    t = text.lower()
    topics = []
    patterns = {
        "work": ["work", "job", "boss", "overtime", "career", "office", "colleague", "fired", "promotion"],
        "relationship": ["partner", "husband", "wife", "boyfriend", "girlfriend", "dating", "dumped", "marriage", "divorce"],
        "family": ["mom", "dad", "mother", "father", "parent", "brother", "sister", "child", "son", "daughter", "family"],
        "grief": ["passed away", "died", "death", "funeral", "gone", "lost someone", "mourning"],
        "health": ["sleep", "tired", "exhausted", "sick", "pain", "hospital", "dying", "cancer"],
        "identity": ["who am i", "purpose", "meaning", "lost", "figure out", "don't know who", "belong"],
        "school": ["school", "class", "teacher", "grade", "homework", "college", "university", "student"],
        "money": ["money", "debt", "poor", "afford", "poverty", "broke", "salary", "pay"],
        "fear": ["afraid", "terrified", "scared", "fear", "danger", "threat", "kill"],
        "freedom": ["trapped", "escape", "prison", "free", "cage", "control", "forced"],
    }
    for topic, keywords in patterns.items():
        if any(k in t for k in keywords):
            topics.append(topic)
    return topics


def detect_tone(text: str) -> str:
    """Detect speaking tone. Zero cost."""
    t = text.lower()
    if any(w in t for w in ["haha", "lol", "joke", "funny", "at least", "consistent right"]):
        return "sarcastic/humorous"
    if any(w in t for w in ["I'm fine", "perfectly", "adequate", "doesn't matter", "whatever"]):
        return "guarded/dismissive"
    if any(w in t for w in ["!", "oh my god", "LOVE", "amazing", "incredible"]):
        return "excited/intense"
    if any(w in t for w in ["I don't know", "what am I", "how to", "supposed to"]):
        return "searching/lost"
    if any(w in t for w in ["they", "against", "nobody", "everyone", "world"]):
        return "adversarial/distrustful"
    if t.count("?") >= 2:
        return "questioning"
    if any(w in t for w in ["I want", "I need", "I must", "I choose"]):
        return "determined"
    return "reflective"


def extract_key_phrases(text: str) -> list[str]:
    """Extract quotable phrases the LLM should reference. Zero cost."""
    phrases = []
    # Quoted-style extraction: short memorable fragments
    patterns = [
        r"(I (?:don'?t|can'?t|won'?t|didn'?t) [a-z ]{3,30})",
        r"((?:my|the) [a-z]+ (?:is|was|keeps?|never|always) [a-z ]{3,25})",
        r"(what (?:am I|is|was|happened|if) [a-z ]{3,20})",
        r"(nobody [a-z ]{3,20})",
        r"(everything [a-z ]{3,20})",
        r"(I (?:just|really|always|never) [a-z ]{3,25})",
    ]
    for p in patterns:
        for m in re.finditer(p, text.lower()):
            phrase = m.group(1).strip()
            if len(phrase.split()) >= 3:
                phrases.append(phrase)
    return phrases[:4]  # max 4


def detect_emotion_words(text: str) -> list[str]:
    """Detect explicit emotion words used. Zero cost."""
    t = text.lower()
    emotion_vocab = {
        "tired", "exhausted", "frustrated", "angry", "sad", "scared", "afraid",
        "terrified", "anxious", "worried", "confused", "lost", "lonely", "alone",
        "happy", "excited", "hopeful", "grateful", "love", "hate", "miss",
        "guilty", "ashamed", "hurt", "broken", "trapped", "suffocating",
        "desperate", "overwhelmed", "empty", "numb", "grief", "mourning",
    }
    found = [w for w in emotion_vocab if w in t.split()]
    return found


def format_precomputed(pc: dict) -> str:
    """Format precomputed signals into minimal text for LLM prompt."""
    parts = []
    if pc["topics"]:
        parts.append(f"Topics: {', '.join(pc['topics'])}")
    parts.append(f"Tone: {pc['tone']}")
    if pc["emotion_words"]:
        parts.append(f"Emotions: {', '.join(pc['emotion_words'])}")
    parts.append(f"Self-ref: {pc['self_ref']:.0%}, Questions: {pc['question_ratio']:.0%}")
    if pc["distress"] > 0.2:
        parts.append(f"Distress: {pc['distress']:.0%}")
    if pc["key_phrases"]:
        parts.append(f"Key phrases: {'; '.join(pc['key_phrases'][:3])}")
    return " | ".join(parts)
