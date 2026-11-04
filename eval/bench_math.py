"""The math script's standard test: the slim detector, no API, a fixed set of AI texts, time and scores together. Window
logits are kept on disk (eval/cache/window_logits.jsonl, by text), so a rerun only pays for text it hasn't seen.

    python -m eval.bench_math
    python -m eval.bench_math --texts 30 --show
    python -m eval.bench_math --band 0.6 0.95 --texts 30
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from eval.gradient_attack import power_attack
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


def log_odds(p: float) -> float:
    """The score without the squeeze near 1, where 0.999 and 0.9999 look alike."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=int, default=12)
    parser.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="only texts of the pool starting between these scores")
    parser.add_argument("--power", action="store_true", help="add the gradient word search")
    parser.add_argument("--show", action="store_true", help="print the first edited text")
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    scorer = CachedScorer(detector, path=LOGITS)
    began = time.time()
    before, after, kept = [], [], Counter()
    edited = ""
    texts = bench_texts(POOL if args.band else args.texts)
    if args.band:
        texts = [text for text in texts if args.band[0] <= scorer(text) <= args.band[1]][:args.texts]
    thresholds = detector.thresholds
    for number, text in enumerate(texts, 1):
        start, runs = time.time(), scorer.runs
        if args.power:
            result = power_attack(text, detector)
            edited, edits, history = result.text, result.kept, result.history
        else:
            edited, edits, history = guided_attack(text, scorer)
        before.append(history[0])
        after.append(history[-1])
        kept.update(edits)
        print(f"{number:>3} {history[0]:.3f} -> {history[-1]:.3f}  log-odds {log_odds(history[0]):+6.2f} -> "
              f"{log_odds(history[-1]):+6.2f}  {len(edits):>2} edits  "
              f"{scorer.runs - runs:>4} model runs  {time.time() - start:.0f}s", flush=True)
    drops = [log_odds(a) - log_odds(b) for a, b in zip(before, after, strict=True)]
    print(f"log-odds drop: mean {statistics.mean(drops):.2f}, median {statistics.median(drops):.2f}; "
          f"left Likely AI (under {thresholds['ai']:.3f}): "
          f"{sum(b > thresholds['ai'] >= a for a, b in zip(after, before, strict=True))}; reached Likely human "
          f"(under {thresholds['review']:.3f}): "
          f"{sum(b > thresholds['review'] >= a for a, b in zip(after, before, strict=True))}; of {len(after)}")
    print(f"mean {statistics.mean(before):.3f} -> {statistics.mean(after):.3f}, median "
          f"{statistics.median(after):.3f}, under 0.5: {sum(p < 0.5 for p in after)} of {len(after)}, "
          f"{time.time() - began:.0f}s, {scorer.runs} model runs, {scorer.hits} cached")
    print(dict(kept))
    if args.show:
        print(edited)


if __name__ == "__main__":
    main()
