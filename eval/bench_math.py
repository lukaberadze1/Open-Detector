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
import sys
import time
from collections import Counter
from pathlib import Path

from eval.gradient_attack import power_attack
from eval.math_attack import GUIDED_BUDGET, CachedScorer, guided_attack
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


class ProgressBar:
    """Counts a scorer's calls and redraws one line: which text, how far through its budget, time so far and left. A
    text can finish before its budget, so the bar may jump at the end of one."""

    def __init__(self, score, total_texts: int, budget: int, width: int = 20) -> None:
        self.score, self.total, self.budget, self.width = score, total_texts, budget, width
        self.text, self.calls = 0, 0
        self.began = time.time()

    def start(self, number: int) -> None:
        self.text, self.calls = number, 0

    def __call__(self, text: str) -> float:
        result = self.score(text)
        self.calls += 1
        self.draw()
        return result

    def draw(self) -> None:
        done = min(1.0, ((self.text - 1) + min(1.0, self.calls / self.budget)) / self.total)
        filled = int(self.width * done)
        spent = time.time() - self.began
        left = spent * (1 - done) / done if done else 0
        print(f"\r[{'#' * filled}{'-' * (self.width - filled)}] {done:4.0%} text {self.text}/{self.total} "
              f"{spent / 60:.0f}m, ~{left / 60:.0f}m left   ",
              end="", file=sys.stderr, flush=True)


INDEPENDENT = "Hello-SimpleAI/chatgpt-detector-roberta"


def independent_scores(texts: list[str], words: int = 300) -> list[float]:
    """A second detector's AI chance for each text, averaged over chunks, since it reads about 512 tokens."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(INDEPENDENT)
    model = AutoModelForSequenceClassification.from_pretrained(INDEPENDENT).eval()
    at = {name.lower(): i for i, name in model.config.id2label.items()}["chatgpt"]
    scores = []
    for text in texts:
        split = text.split()
        chunks = [" ".join(split[i:i + words]) for i in range(0, len(split), words)]
        encoded = tokenizer(chunks, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            scores.append(float(model(**encoded).logits.softmax(-1)[:, at].mean()))
    return scores


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
    parser.add_argument("--contextual", action="store_true", help="add the model-backed word search")
    parser.add_argument("--paraphraser", choices=("t5", "pegasus", "dipper"),
                        help="with --contextual, rewrite sentences first")
    parser.add_argument("--out", type=Path, help="save each text before and after, labelled AI, as .jsonl")
    parser.add_argument("--independent", action="store_true", help="also score with a detector the search never sees")
    parser.add_argument("--show", action="store_true", help="print the first edited text")
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    scorer = CachedScorer(detector, path=LOGITS)
    began = time.time()
    before, after, kept = [], [], Counter()
    pairs: list[tuple[str, str]] = []
    edited = ""
    texts = bench_texts(POOL if args.band else args.texts)
    if args.band:
        texts = [text for text in texts if args.band[0] <= scorer(text) <= args.band[1]][:args.texts]
    thresholds = detector.thresholds
    bar = ProgressBar(scorer, len(texts), 400 if args.contextual else GUIDED_BUDGET)
    for number, text in enumerate(texts, 1):
        start, runs = time.time(), scorer.runs
        bar.start(number)
        if args.contextual:
            from eval.contextual_attack import MeaningGuard, Quality, contextual_attack, paraphrase_stage

            guard = MeaningGuard()
            start_text, prior = text, []
            if args.paraphraser:
                stage = paraphrase_stage(text, bar, guard, quality=Quality(), name=args.paraphraser)
                start_text, prior = stage.text, stage.history
            result = contextual_attack(start_text, detector, bar, guard)
            edited, history = result.text, [*prior, *result.history]
            edits = ["swap"] * len(result.swaps)
            if args.show:
                print("   ", ", ".join(f"{a}>{b}" for a, b in result.swaps), flush=True)
        elif args.power:
            result = power_attack(text, detector)
            edited, edits, history = result.text, result.kept, result.history
        else:
            edited, edits, history = guided_attack(text, bar)
        pairs.append((text, edited))
        before.append(history[0])
        after.append(history[-1])
        kept.update(edits)
        print("\r", end="", file=sys.stderr)
        print(f"{number:>3} {history[0]:.3f} -> {history[-1]:.3f}  log-odds {log_odds(history[0]):+6.2f} -> "
              f"{log_odds(history[-1]):+6.2f}  {len(edits):>2} edits  "
              f"{scorer.runs - runs:>4} model runs  {time.time() - start:.0f}s", flush=True)
    drops = [log_odds(a) - log_odds(b) for a, b in zip(before, after, strict=True)]
    print(file=sys.stderr)
    print(f"log-odds drop: mean {statistics.mean(drops):.2f}, median {statistics.median(drops):.2f}; "
          f"left Likely AI (under {thresholds['ai']:.3f}): "
          f"{sum(b > thresholds['ai'] >= a for a, b in zip(after, before, strict=True))}; reached Likely human "
          f"(under {thresholds['review']:.3f}): "
          f"{sum(b > thresholds['review'] >= a for a, b in zip(after, before, strict=True))}; of {len(after)}")
    print(f"mean {statistics.mean(before):.3f} -> {statistics.mean(after):.3f}, median "
          f"{statistics.median(after):.3f}, under 0.5: {sum(p < 0.5 for p in after)} of {len(after)}, "
          f"{time.time() - began:.0f}s, {scorer.runs} model runs, {scorer.hits} cached")
    print(dict(kept))
    held = independent_scores([t for pair in pairs for t in pair]) if args.independent else []
    if held:
        print("independent detector (mean AI share of its windows; the search never saw it):")
        for number, (was, now) in enumerate(zip(held[::2], held[1::2], strict=True), 1):
            print(f"{number:>3} {was:.3f} -> {now:.3f}")
        print(f"mean {statistics.mean(held[::2]):.3f} -> {statistics.mean(held[1::2]):.3f}")
    if args.out:
        with args.out.open("a", encoding="utf-8") as file:
            for i, (old, new) in enumerate(pairs):
                attack = "math+contextual" + (f"+{args.paraphraser}" if args.paraphraser else "")
                file.write(json.dumps({"text": new, "label": 1, "model": "math-contextual", "source_text": old,
                                       "attack": attack,
                                       "score_before": before[i], "score_after": after[i],
                                       **({"independent_before": held[2 * i], "independent_after": held[2 * i + 1]}
                                          if held else {})}, ensure_ascii=False) + "\n")
    if args.show:
        print(edited)


if __name__ == "__main__":
    main()
