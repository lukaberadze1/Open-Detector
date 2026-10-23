"""Model-free style signals: sentence-length burstiness, vocabulary spread, and AI-typical phrasing.

These are the cheap half of what commercial detectors say they measure. Burstiness is how much
sentence length varies: people follow a 40-word sentence with a 3-word one, while raw LLM output
keeps an even rhythm. AI-isms are words and phrases that LLMs use far more often than people do
("delve", "a rich tapestry", "it's worth noting"). Kobak et al. (2024) measured the jump in these
words in PubMed abstracts after ChatGPT's release. Vocabulary spread (moving-average type-token
ratio) rises when a word spinner swaps common words for rarer synonyms.

None of these decides anything alone. They are combiner features: the combiner learns how much each
is worth, and thresholds are still set on held-out human text, so the overall false-positive target
holds. Check the per-group slices of the report anyway, because a phrase list can hit some writers
harder than others.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .segment import Sentence

FEATURES = ("style_len_cv", "style_mattr", "style_aiisms")

MIN_SENTENCES = 4  # fewer than this and sentence-length spread isn't measurable
MATTR_WINDOW = 50

# Words and phrases LLMs overuse. Each is distinctive on its own; generic academic connectives
# ("furthermore", "in conclusion") are left out on purpose, because students use them all the time.
_AI_PATTERNS = [
    r"delv(?:e|es|ed|ing)",
    r"tapestr(?:y|ies)",
    r"testament to",
    r"(?:in|into|within|beyond) the realms? of",
    r"underscor(?:e|es|ed|ing)",
    r"showcas(?:e|es|ed|ing)",
    r"pivotal",
    r"intricac(?:y|ies)",
    r"intricate(?:ly)?",
    r"meticulous(?:ly)?",
    r"multifaceted",
    r"commendable",
    r"noteworthy",
    r"paramount",
    r"(?:ever-evolving|evolving|ever-changing|digital|modern|competitive|complex) landscape",
    r"ever-(?:evolving|changing|growing|expanding)",
    r"it(?:'s|’s| is) (?:worth noting|important to note|crucial to note|essential to note)",
    r"plays? an? (?:crucial|pivotal|vital) role",
    r"in today(?:'s|’s) (?:fast-paced|digital|modern|ever-changing|rapidly changing|interconnected) "
    r"(?:world|age|society|era)",
    r"navigat(?:e|es|ed|ing) the (?:complexities|intricacies|nuances)",
    r"embark(?:s|ed|ing)? on (?:a|an|this|the) (?:journey|quest)",
    r"foster(?:s|ed|ing)? (?:a|an) (?:sense|culture|environment|spirit|atmosphere)",
    r"garner(?:s|ed|ing)?",
    r"unwavering",
    r"harness(?:es|ed|ing)? the (?:power|potential)",
    r"seamless(?:ly)?",
    r"(?:a|the) beacon of",
    r"myriad",
    r"resonat(?:e|es|ed|ing) (?:with|deeply)",
    r"shed(?:s|ding)? (?:new )?light on",
    r"serv(?:e|es|ed|ing) as an? (?:stark |powerful |poignant |vivid )?reminder",
    r"delicate balance",
    r"symphony of",
    r"nuanced (?:understanding|approach|perspective)",
    r"holistic(?:ally)?",
    r"indelible (?:mark|impression)",
    r"(?:intricate|complex|dynamic) interplay",
    r"unlock(?:s|ed|ing)? (?:the|new|its|their) (?:full )?potential",
]
_AI_RE = re.compile(r"\b(?:" + "|".join(_AI_PATTERNS) + r")\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z]+(?:['’][a-z]+)?")
_TERMINAL_RE = re.compile(r"[.!?…][\"'”’)\]]*$")


@dataclass(frozen=True)
class Phrase:
    start: int
    end: int
    text: str


def ai_phrases(text: str) -> list[Phrase]:
    """Every AI-typical word or phrase in `text`, with character offsets."""
    return [Phrase(m.start(), m.end(), m.group(0)) for m in _AI_RE.finditer(text)]


def phrase_counts(text: str) -> Counter:
    """How often each AI-typical phrase occurs, case-folded, most frequent first."""
    return Counter(p.text.lower().replace("’", "'") for p in ai_phrases(text))


def sentence_length_cv(text: str, sentences: Sequence[Sentence]) -> float:
    """Coefficient of variation of sentence lengths in words: the burstiness of the rhythm.
    Only full sentences count. Headings and list items, which have no closing punctuation, are
    left out, but a two-word sentence like "We left." stays: short ones are what make it bursty."""
    lengths = [s.words for s in sentences if _TERMINAL_RE.search(text[s.start : s.end])]
    if len(lengths) < MIN_SENTENCES:
        return math.nan
    mean = sum(lengths) / len(lengths)
    var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(var) / mean


def mattr(text: str, window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio (Covington and McFall, 2010). Unlike a plain type-token
    ratio, it doesn't fall as texts get longer, so short and long essays are comparable."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else math.nan
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, len(tokens)):
        old, new = tokens[i - window], tokens[i]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[new] += 1
        total += len(counts)
    return total / (len(tokens) - window + 1) / window


def features(text: str, sentences: Sequence[Sentence]) -> dict[str, float]:
    """The style features of the given sentences of `text` (all of them for a document, or the
    ones a window covers)."""
    body = " ".join(text[s.start : s.end] for s in sentences)
    words = sum(s.words for s in sentences)
    hits = len(_AI_RE.findall(body))
    return {
        "style_len_cv": sentence_length_cv(text, sentences),
        "style_mattr": mattr(body),
        # log1p of hits per 1,000 words: the first few hits say much more than the twentieth.
        "style_aiisms": math.log1p(1000 * hits / words) if words else math.nan,
    }
