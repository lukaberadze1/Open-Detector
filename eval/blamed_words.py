"""Which words lower the slim detector's score most when masked across the benchmark's AI texts, and whether the math
script has a safe replacement for them: the list to write new `SAFE_SYNONYMS` from. No API.

    python -m eval.blamed_words --texts 40
"""

from __future__ import annotations

import argparse
from collections import Counter

from eval.bench_math import bench_texts
from eval.gradient_attack import _WORD, masked_pushes, word_pushes
from eval.math_attack import SAFE_SYNONYMS
from opendetector.normalize import normalize
from opendetector.segment import sentence_windows, split_sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=int, default=40)
    parser.add_argument("--top", type=int, default=150)
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    engine = next(c for c in detector.classifiers if hasattr(c, "embedded_log_odds"))
    config = detector.config
    blame, seen = Counter(), Counter()
    for text in bench_texts(args.texts):
        clean = normalize(text)[0]
        windows = sentence_windows(split_sentences(clean), config.doc_window_words, config.doc_stride_words)
        for window in windows[:1]:  # hundreds of model runs a window
            chunk = clean[window.start:window.end]
            spans = [match.span() for match in _WORD.finditer(chunk)]
            for push, start, end in word_pushes(chunk, masked_pushes(engine, chunk, spans)):
                word = chunk[start:end].lower()
                blame[word] += push
                seen[word] += 1
    for word, push in blame.most_common(args.top):
        print(f"{word:<18} {push:8.2f} {seen[word]:>4}x  {'covered' if word in SAFE_SYNONYMS else ''}")
    covered = sum(push for word, push in blame.items() if word in SAFE_SYNONYMS)
    print(f"safe list covers {covered / sum(blame.values()):.1%} of the blame")


if __name__ == "__main__":
    main()
