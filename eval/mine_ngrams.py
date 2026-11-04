"""Which 3-grams and 4-grams are over-represented in AI text against human text, by how many documents hold them: the
evidence behind the math script's phrase rules in opendetector/core/rules/. Uses MAGE's test split in the Hugging Face
cache and writes eval/reports/ngrams.md. The rules' replacements are written by hand from this list, since a count
can't say what a phrase should become.

    python -m eval.mine_ngrams
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from eval.datasets import load_mage

REPORT = Path(__file__).resolve().parent / "reports/ngrams.md"
# MAGE runs many generators over one source, so a topic's words ('yellow cabs') fill many AI documents and no human
# ones. A phrase has to be common in both to be a matter of style.
MIN_AI_DOCS = 200
MIN_HUMAN_DOCS = 20
TOP = 150
_WORD = re.compile(r"[a-z']+")


def document_counts(texts) -> Counter:
    """In how many of `texts` each 3-gram and 4-gram occurs, not crossing a sentence's end."""
    counts: Counter = Counter()
    for text in texts:
        seen = set()
        for sentence in re.split(r"[.!?\n]+", text.lower()):
            words = _WORD.findall(sentence)
            for size in (3, 4):
                seen.update(" ".join(words[i:i + size]) for i in range(len(words) - size + 1))
        counts.update(seen)
    return counts


def ranked(ai: Counter, human: Counter, ai_docs: int, human_docs: int, min_ai: int = MIN_AI_DOCS) -> list[tuple]:
    """(phrase, log-odds of AI over human with add-one smoothing, AI documents, human documents), highest first."""
    rows = []
    for phrase, count in ai.items():
        other = human.get(phrase, 0)
        if count >= min_ai and other >= MIN_HUMAN_DOCS:
            rows.append((phrase, math.log((count + 1) / (ai_docs - count + 1))
                         - math.log((other + 1) / (human_docs - other + 1)), count, other))
    return sorted(rows, key=lambda row: -row[1])


def main() -> None:
    frame = load_mage(n=10**7)
    ai_texts, human_texts = frame[frame.label == 1].text, frame[frame.label == 0].text
    rows = ranked(document_counts(ai_texts), document_counts(human_texts), len(ai_texts), len(human_texts))
    lines = ["# N-grams over-represented in AI text", "",
             f"MAGE test split: {len(ai_texts):,} AI and {len(human_texts):,} human documents. Written by "
             "`eval/mine_ngrams.py`. A document counts once per phrase.", "",
             "| Phrase | Log-odds | AI documents | Human documents |", "|---|---|---|---|"]
    lines += [f"| {phrase} | {odds:.2f} | {ai:,} | {human:,} |" for phrase, odds, ai, human in rows[:TOP]]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows):,} phrases ranked; top {TOP} in {REPORT}")


if __name__ == "__main__":
    main()
