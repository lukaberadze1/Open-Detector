"""Do the math script's drops, found against the slim detector, hold on the full build? Edits the benchmark's borderline
texts guided by slim alone, then scores each text before and after on the full build, one build in memory at a time.
No API; needs about 3.5 GB for the full build.

    python -m eval.math_transfer
"""

from __future__ import annotations

import argparse
import gc
import statistics

from eval.bench_math import LOGITS, POOL, bench_texts, log_odds
from eval.math_attack import CachedScorer, guided_attack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--band", type=float, nargs=2, default=(0.6, 0.95), metavar=("LOW", "HIGH"))
    parser.add_argument("--texts", type=int, default=30)
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    slim = Detector.from_config(DetectorConfig.for_profile("slim"))
    scorer = CachedScorer(slim, path=LOGITS)
    texts = [t for t in bench_texts(POOL) if args.band[0] <= scorer(t) <= args.band[1]][:args.texts]
    pairs = []
    for text in texts:
        edited, _, history = guided_attack(text, scorer)
        pairs.append((text, edited, history[0], history[-1]))
    del slim, scorer
    gc.collect()

    full = Detector.from_config(DetectorConfig())
    ai_line = full.thresholds["ai"]
    rows = []
    for number, (text, edited, slim_before, slim_after) in enumerate(pairs, 1):
        before, after = full.score(text), full.score(edited)
        rows.append((slim_before, slim_after, before, after))
        print(f"{number:>3} slim {slim_before:.3f} -> {slim_after:.3f}   full {before:.3f} -> {after:.3f}", flush=True)
    for name, b, a in (("slim", 0, 1), ("full", 2, 3)):
        drops = [log_odds(row[b]) - log_odds(row[a]) for row in rows]
        print(f"{name}: mean {statistics.mean(row[b] for row in rows):.3f} -> "
              f"{statistics.mean(row[a] for row in rows):.3f}, median log-odds drop {statistics.median(drops):.2f}")
    print(f"full build: {sum(row[2] > ai_line >= row[3] for row in rows)} of "
          f"{sum(row[2] > ai_line for row in rows)} Likely AI texts left that verdict (line {ai_line:.3f})")


if __name__ == "__main__":
    main()
