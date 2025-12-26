"""Readability / grading helpers.

This module is intentionally streamlit-free so it can be unit-tested in isolation.
The app imports equivalent logic for runtime use.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_VOWELS = "aeiouyàâäæçéèêëîïôöœùûüÿ"


def _count_sentences(text: str) -> int:
    if not text:
        return 1
    s = re.split(r"[.!?;]+|\n+", text)
    n = sum(1 for part in s if part.strip())
    return max(1, n)


def _count_words(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", text or "")))


def _count_syllables_word(word: str) -> int:
    w = (word or "").lower()
    if not w:
        return 1
    w = re.sub(r"[^a-zà-öø-ÿ]", "", w)
    if not w:
        return 1
    groups = re.findall(rf"[{_VOWELS}]+", w)
    syl = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ye")) and syl > 1:
        syl -= 1
    return max(1, syl)


def _count_syllables(text: str) -> int:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", text or "")
    return sum(_count_syllables_word(w) for w in words) or 1


def flesch_kincaid(text: str) -> float:
    """Flesch–Kincaid Grade Level (English). Lower is easier."""
    words = _count_words(text)
    sentences = _count_sentences(text)
    syllables = _count_syllables(text)
    grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    return round(max(0.0, min(20.0, float(grade))), 2)


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text.strip()) if p.strip()]


def per_sentence_grades_en(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in split_sentences(text):
        out.append({"sentence": s, "grade": flesch_kincaid(s)})
    return out


def worst_sentences_en(text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    rows = per_sentence_grades_en(text)
    rows.sort(key=lambda r: float(r.get("grade", 0.0) or 0.0), reverse=True)
    return rows[: max(0, int(top_n))]
