"""Rebuild the math script's table of word pairs from MAGE's human texts in the Hugging Face cache: counts of pairs seen
3+ times, no text. Kept apart from eval/math_attack.py, which the downloadable app ships without the dataset loader.

    python -m eval.build_bigrams
"""

from __future__ import annotations

import gzip
import itertools
import json
from collections import Counter

from eval.datasets import load_mage
from eval.math_attack import _TOKEN, BIGRAMS

LIMIT = 400_000


def main() -> None:
    frame = load_mage(n=10**7)  # the whole split: its default is a 2,000-text sample
    counts: Counter = Counter()
    for text in frame[frame.label == 0].text:
        counts.update(f"{a} {b}" for a, b in itertools.pairwise(_TOKEN.findall(text.lower())))
    kept = dict(sorted(((k, v) for k, v in counts.items() if v >= 3), key=lambda kv: -kv[1])[:LIMIT])
    with gzip.open(BIGRAMS, "wt", encoding="utf-8") as file:
        json.dump(kept, file)
    print(f"{len(kept):,} pairs written to {BIGRAMS}")


if __name__ == "__main__":
    main()
