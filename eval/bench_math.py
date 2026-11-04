"""The math script's standard test: the slim detector, no API, a fixed set of AI texts, time and scores together. Window
logits are kept on disk (eval/cache/window_logits.jsonl, by text), so a rerun only pays for text it hasn't seen.

    python -m eval.bench_math
    python -m eval.bench_math --texts 30 --show
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from eval.math_attack import CachedScorer, guided_attack
from opendetector.segment import count_words

CACHE = Path(__file__).resolve().parent / "cache"
SAMPLES = CACHE / "bench_texts.jsonl"
LOGITS = CACHE / "window_logits.jsonl"
POOL = 200


def bench_texts(count: int) -> list[str]:
    """AI texts of 150+ words from MAGE, saved once so a run needn't load the dataset."""
    if not SAMPLES.exists():
        from eval.datasets import load_mage

        frame = load_mage()
        frame = frame[frame.label == 1]
        frame = frame[frame.text.map(count_words) >= 150]
        chosen = frame.sample(min(POOL, len(frame)), random_state=7).text
        CACHE.mkdir(parents=True, exist_ok=True)
        SAMPLES.write_text("".join(json.dumps(text) + "\n" for text in chosen), encoding="utf-8")
    return [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines()][:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=int, default=12)
    parser.add_argument("--show", action="store_true", help="print the first edited text")
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    scorer = CachedScorer(detector, path=LOGITS)
    began = time.time()
    before, after, kept = [], [], Counter()
    edited = ""
    for number, text in enumerate(bench_texts(args.texts), 1):
        start, runs = time.time(), scorer.runs
        edited, edits, history = guided_attack(text, scorer)
        before.append(history[0])
        after.append(history[-1])
        kept.update(edits)
        print(f"{number:>3} {history[0]:.3f} -> {history[-1]:.3f}  {len(edits):>2} edits  "
              f"{scorer.runs - runs:>4} model runs  {time.time() - start:.0f}s", flush=True)
    print(f"mean {statistics.mean(before):.3f} -> {statistics.mean(after):.3f}, median "
          f"{statistics.median(after):.3f}, under 0.5: {sum(p < 0.5 for p in after)} of {len(after)}, "
          f"{time.time() - began:.0f}s, {scorer.runs} model runs, {scorer.hits} cached")
    print(dict(kept))
    if args.show:
        print(edited)


if __name__ == "__main__":
    main()
