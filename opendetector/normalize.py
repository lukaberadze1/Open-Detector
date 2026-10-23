"""Input normalization that undoes cheap evasion tricks before any model sees the text.

RAID's attack suite includes zero-width characters, homoglyphs and whitespace padding.
They change nothing a reader sees but scramble tokenization, so we strip them first and
report how many we found: hidden characters in an essay are evidence in their own right.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Invisible characters that have no business in prose.
_INVISIBLE = "​‌‍‎‏⁠⁡⁢⁣⁤﻿­᠎͏"
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE}]")

# Cyrillic and Greek letters that render identically to Latin ones.
_CONFUSABLES = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w", "ӏ": "l",
    # Cyrillic uppercase
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "Ԁ": "D",
    # Greek
    "ο": "o", "ν": "v", "ρ": "p", "ι": "i", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Χ": "X", "Υ": "Y",
}
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_LINE_PAD_RE = re.compile(r" *\n *")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NormalizationReport:
    invisible_chars: int = 0
    homoglyphs: int = 0
    compatibility_chars: int = 0

    @property
    def suspicious(self) -> bool:
        return self.invisible_chars > 0 or self.homoglyphs > 0


def _is_latin_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _fix_homoglyphs(text: str) -> tuple[str, int]:
    letters = [ch for ch in text if ch.isalpha()]
    mostly_latin = bool(letters) and sum(map(_is_latin_letter, letters)) / len(letters) > 0.9
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        word = match.group(0)
        confusable = sum(ch in _CONFUSABLES for ch in word)
        if not confusable:
            return word
        has_latin = any(map(_is_latin_letter, word))
        all_confusable = all(ch in _CONFUSABLES or not ch.isalpha() for ch in word)
        # Mixed-script words are always an attack; all-lookalike words only count as one
        # inside Latin text, so genuine Russian or Greek passages survive untouched.
        if has_latin or (mostly_latin and all_confusable):
            count += confusable
            return "".join(_CONFUSABLES.get(ch, ch) for ch in word)
        return word

    return _WORD_RE.sub(repl, text), count


def normalize(text: str) -> tuple[str, NormalizationReport]:
    invisible = len(_INVISIBLE_RE.findall(text))
    text = _INVISIBLE_RE.sub("", text)

    compat = sum(unicodedata.normalize("NFKC", ch) != ch for ch in text)
    text = unicodedata.normalize("NFKC", text)

    text, homoglyphs = _fix_homoglyphs(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_RE.sub(" ", text)
    text = _LINE_PAD_RE.sub("\n", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip(), NormalizationReport(invisible, homoglyphs, compat)
