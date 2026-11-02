"""Word swaps for the built-in human profile only, run on every reply as it is cleaned: stock phrases and words an
AI favours for plainer ones, from styles/human_default/swaps.json (ported from ClearRewriter), and contractions. The
built-in profile describes no one in particular, so these can't pull a text away from a writer. They would from a
user's own profile, so they stay off there, and off in public mode. Each text picks its swaps the same way every
time, seeded by the text. Pure Python.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path

SWAPS = Path(__file__).resolve().parents[1] / "styles/human_default/swaps.json"
SYNONYM_RATE = 0.15  # of the words with a synonym: more reads as a thesaurus
CONTRACTION_RATE = 0.5
# Words whose listed synonyms don't fit every use: 'before the war' can't become 'earlier the war'.
RISKY = {"before", "after", "still", "right", "about", "over", "like", "clear", "change", "show", "work", "way"}
_CONTRACTIONS = {"do not": "don't", "does not": "doesn't", "did not": "didn't", "cannot": "can't", "can not": "can't",
                 "will not": "won't", "would not": "wouldn't", "should not": "shouldn't", "could not": "couldn't",
                 "is not": "isn't", "are not": "aren't", "was not": "wasn't", "were not": "weren't",
                 "has not": "hasn't", "have not": "haven't", "had not": "hadn't", "it is": "it's",
                 "they are": "they're", "that is": "that's", "there is": "there's", "it will": "it'll"}
_SENTENCE_START = re.compile(r"(^|(?<![.][a-z])(?<!\b[A-Z])[.!?]\s+|\n\s*)([a-z])")


@lru_cache(maxsize=1)
def _tables() -> tuple:
    data = json.loads(SWAPS.read_text(encoding="utf-8"))
    def pattern(words):
        return re.compile(r"\b(" + "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True)) + r")\b",
                          re.I)
    synonyms = {word: choices for word, choices in data["synonyms"].items() if word not in RISKY}
    lexicon = [(re.compile(source, re.I), target) for source, target in data["lexicon"]]
    return data["phrases"], pattern(data["phrases"]), synonyms, pattern(synonyms), lexicon, pattern(_CONTRACTIONS)


def _like(found: str, new: str) -> str:
    return new[:1].upper() + new[1:] if found[:1].isupper() else new


def human_swaps(text: str) -> str:
    """`text` with stock phrases, then lexicon words, then some synonyms and contractions swapped."""
    if not text.strip():
        return text
    rng = random.Random(hashlib.sha256(text.encode()).hexdigest())
    phrases, phrase_re, synonyms, synonym_re, lexicon, contraction_re = _tables()
    out = phrase_re.sub(lambda m: _like(m.group(), rng.choice(phrases[m.group().lower()])), text)
    for pattern, target in lexicon:
        out = pattern.sub(lambda m, target=target: _like(m.group(), target) if target else "", out)

    def synonym(match: re.Match) -> str:
        choices = synonyms[match.group().lower()]
        return _like(match.group(), rng.choice(choices)) if rng.random() < SYNONYM_RATE else match.group()

    out = synonym_re.sub(synonym, out)
    out = contraction_re.sub(lambda m: _like(m.group(), _CONTRACTIONS[m.group().lower()])
                             if rng.random() < CONTRACTION_RATE else m.group(), out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r",\s*\.", ".", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), out)
