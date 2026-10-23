"""Sentence splitting with character spans, and sentence-aligned sliding windows.

Detectors are unreliable on single sentences, so nothing is ever scored alone: text is
cut into overlapping windows of whole sentences, each window is scored, and a sentence
inherits the average of every window that covers it (the approach Turnitin and Pangram
describe). Windows are sized in words so every engine sees the same windows regardless
of its tokenizer.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "al",
    "fig", "figs", "no", "vol", "pp", "p", "ch", "sec", "eq", "approx", "dept", "inc",
    "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "u.s", "u.k", "a.m", "p.m", "ph.d", "cf", "ca", "ed", "eds",
}
# Sentence-ending punctuation, optional closing quotes/brackets, then whitespace.
_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    words: int


@dataclass(frozen=True)
class Window:
    first: int  # index of the first sentence
    last: int  # index of the last sentence, inclusive
    start: int  # character offsets into the text
    end: int
    words: int


def _is_abbreviation(text: str, punct_start: int) -> bool:
    if text[punct_start] != ".":
        return False
    word_start = punct_start
    while word_start > 0 and not text[word_start - 1].isspace() and text[word_start - 1] not in "(\"'“‘[":
        word_start -= 1
    token = text[word_start:punct_start].lower()
    if not token:
        return False
    # Single initials ("J. Smith") and listed abbreviations don't end sentences.
    return (len(token) == 1 and token.isalpha()) or token in _ABBREVIATIONS


def _starts_sentence(text: str, pos: int) -> bool:
    while pos < len(text) and text[pos].isspace():
        if text[pos] == "\n":
            return True
        pos += 1
    if pos >= len(text):
        return True
    ch = text[pos]
    return ch.isupper() or ch.isdigit() or ch in "\"'“‘([" or not ch.isalpha()


def split_sentences(text: str) -> list[Sentence]:
    cuts = [0]
    for line in re.finditer(r"\n", text):
        cuts.append(line.end())
    for match in _BOUNDARY_RE.finditer(text):
        if _is_abbreviation(text, match.start()) or not _starts_sentence(text, match.end()):
            continue
        cuts.append(match.end())
    cuts.append(len(text))
    bounds = sorted(set(cuts))

    sentences = []
    for a, b in itertools.pairwise(bounds):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            sentences.append(Sentence(a, b, len(_WORD_RE.findall(text[a:b]))))
    return sentences


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def sentence_windows(sentences: list[Sentence], target_words: int, stride_words: int) -> list[Window]:
    """Group consecutive sentences into windows of about `target_words`, advancing about
    `stride_words` each step. Every sentence is covered, and the final window is anchored
    to the end of the text so the tail is scored with full context."""
    n = len(sentences)
    if n == 0:
        return []
    counts = [max(s.words, 1) for s in sentences]

    def make(i: int, j: int) -> Window:
        return Window(i, j, sentences[i].start, sentences[j].end, sum(counts[i : j + 1]))

    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        j, total = i, counts[i]
        while j + 1 < n and total + counts[j + 1] <= target_words:
            j += 1
            total += counts[j]
        if j == n - 1:
            while i > 0 and total + counts[i - 1] <= target_words:
                i -= 1
                total += counts[i]
            if not spans or spans[-1] != (i, j):
                spans.append((i, j))
            break
        spans.append((i, j))
        advanced, k = 0, i
        while k <= j and advanced < stride_words:
            advanced += counts[k]
            k += 1
        i = max(k, i + 1)
    return [make(i, j) for i, j in spans]
