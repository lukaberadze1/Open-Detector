"""Sentence splitting with character spans, and sentence-aligned sliding windows.

Detectors are unreliable on single sentences, so nothing is ever scored alone: text is
cut into overlapping windows of whole sentences, each window is scored, and a sentence
inherits the average of every window that covers it (the approach Turnitin and Pangram
describe). Windows are sized in words so every engine sees the same windows regardless
of its tokenizer.
"""

from __future__ import annotations

import bisect
import itertools
import re
import statistics
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
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
_LINE_END = (".", "!", "?", ":", ";", '"', "”", ")")


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    words: int
    paragraph: int = 0  # index of the blank-line-separated paragraph it sits in


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
    paragraph_breaks = [m.start() for m in _PARAGRAPH_BREAK_RE.finditer(text)]

    sentences = []
    for a, b in itertools.pairwise(bounds):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            paragraph = bisect.bisect_left(paragraph_breaks, a)
            sentences.append(Sentence(a, b, len(_WORD_RE.findall(text[a:b])), paragraph))
    return sentences


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _unwrap(lines: list[str]) -> list[str]:
    """Rejoin hard-wrapped prose: a line that fills the width (45+ characters), doesn't end a
    sentence, and is continued in lowercase. Verse lines are usually shorter than that, so
    enjambed lowercase poems are not mistaken for wrapped prose."""
    out: list[str] = []
    previous = ""
    for line in lines:
        if out and len(previous) >= 45 and not out[-1].endswith(_LINE_END) and line[:1].islower():
            out[-1] += " " + line
        else:
            out.append(line)
        previous = line
    return out


def looks_like_prose(text: str, min_lines: int = 6, max_median_words: int = 10) -> bool:
    """False for verse, lists and similar line-by-line formats, which the detector isn't built
    for (Turnitin doesn't score poetry, scripts or lists either). On RAID's 150+ word texts this
    flags 84% of human poems and 92% of human recipes, but 0-2% of human prose in other domains."""
    lines = _unwrap([line.strip() for line in text.splitlines() if line.strip()])
    if len(lines) < min_lines:
        return True
    return statistics.median(len(line.split()) for line in lines) > max_median_words


def paragraph_blocks(sentences: list[Sentence], min_words: int) -> list[tuple[int, int]]:
    """Sentence-index ranges (first, last) of paragraphs. A paragraph shorter than `min_words`
    (a heading, a one-liner) is merged into the next one, or into the previous one at the
    end, so every block is long enough to score reliably."""
    blocks: list[tuple[int, int]] = []
    start, words = 0, 0
    for i, s in enumerate(sentences):
        words += s.words
        paragraph_ends = i + 1 == len(sentences) or sentences[i + 1].paragraph != s.paragraph
        if paragraph_ends and words >= min_words:
            blocks.append((start, i))
            start, words = i + 1, 0
    if start < len(sentences):  # a short tail joins the block before it
        blocks[-1:] = [(blocks[-1][0] if blocks else start, len(sentences) - 1)]
    return blocks


def sentence_windows(sentences: list[Sentence], target_words: int, stride_words: int,
                     min_paragraph_words: int | None = None) -> list[Window]:
    """Group consecutive sentences into windows of about `target_words`, advancing about
    `stride_words` each step. Every sentence is covered, and the final window is anchored
    to the end of the text so the tail is scored with full context.

    With `min_paragraph_words`, windows never cross a paragraph break (see paragraph_blocks),
    so a sentence isn't scored together with a differently written paragraph next to it."""
    if min_paragraph_words is None:
        return _windows(sentences, 0, len(sentences) - 1, target_words, stride_words)
    return [w for first, last in paragraph_blocks(sentences, min_paragraph_words)
            for w in _windows(sentences, first, last, target_words, stride_words)]


def _windows(sentences: list[Sentence], lo: int, hi: int, target_words: int, stride_words: int) -> list[Window]:
    """The sliding windows over sentences[lo..hi] (inclusive), with global indices."""
    n = hi - lo + 1
    if n <= 0:
        return []
    counts = [max(s.words, 1) for s in sentences[lo : hi + 1]]

    def make(i: int, j: int) -> Window:
        return Window(lo + i, lo + j, sentences[lo + i].start, sentences[lo + j].end, sum(counts[i : j + 1]))

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
