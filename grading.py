import re
from typing import List

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text.strip()) if p.strip()]

def _count_sentences(text: str) -> int:
    if not text:
        return 1
    parts = re.split(r"[.!?;]+|\n+", text)
    n = sum(1 for p in parts if p.strip())
    return max(1, n)

def _count_words(text: str) -> int:
    if not text:
        return 1
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text, flags=re.UNICODE)
    return max(1, len(words))

def _count_syllables_word(word: str) -> int:
    w = (word or "").lower()
    if not w:
        return 1
    w = "".join(ch for ch in w if ch.isalpha())
    if not w:
        return 1
    vowels = set("aeiouyàâäæéèêëîïôöœùûüÿ")
    syl = 0
    prev = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev:
            syl += 1
        prev = is_v
    if w.endswith("e") and not w.endswith(("le","ye")) and syl > 1:
        syl -= 1
    return max(1, syl)

def _count_syllables(text: str) -> int:
    if not text:
        return 1
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text, flags=re.UNICODE)
    return sum(_count_syllables_word(w) for w in words) or 1

def flesch_kincaid(text: str) -> float:
    words = _count_words(text)
    sentences = _count_sentences(text)
    syllables = _count_syllables(text)
    grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    return round(max(0.0, min(20.0, float(grade))), 2)
