"""Grades the slim detector on humanize-text's demo data (lynote-ai/humanize-text, training-data-pipeline/data/demos):
50 AI texts, the human rewrite of each, and six rewrites of each by their own model at temperatures from 0.3 to 1.0.
It's a free look at what a trained humanizer does to our detector. Scores come from the math benchmark's cache, so a
stopped run carries on. Read-only: the texts stay in gitignored data/pairs/humanize_text_demo/, since the demo's license
is unclear, and aren't training data. One of the 15 'private' human rewrites opens "Here is a paraphrased version", so
treat the human row as a rough false-positive reading.

    python -m eval.score_demo --report eval/reports/humanize_demo.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

from opendetector.segment import count_words

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data/pairs/humanize_text_demo"
MIN_WORDS = 150  # the detector abstains below this
AI_SOURCE, HUMAN = "AI source", "Human rewrite"


def demo_groups(folder: Path) -> dict[str, list[str]]:
    """The texts by what they are, each rewrite under its model's temperature, 150+ words only."""
    def rows(name: str) -> list[dict]:
        return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]

    groups: dict[str, list[str]] = {AI_SOURCE: [], HUMAN: []}
    for seed in rows("kto_seed_demo.jsonl"):
        groups[AI_SOURCE].append(seed["messages"][1]["content"])
        groups[HUMAN].append(seed["desirable_response"])
    for candidate in rows("kto_candidates_demo.jsonl"):
        for result in candidate["results"]:
            groups.setdefault(f"Their model, temperature {result['temperature']:.1f}", []).append(result["rewrite"])
    return {name: [text for text in texts if count_words(text) >= MIN_WORDS] for name, texts in groups.items()}


def report(groups: dict[str, list[str]], score: Callable[[str], float], thresholds: dict[str, float],
           progress: Callable[[str], None] = lambda line: None) -> str:
    lines = ["# humanize-text's demo data against the slim detector", "",
             f"Likely AI is above {thresholds['ai']:.3f}, Likely human at or under {thresholds['review']:.3f}. Written "
             "by `python -m eval.score_demo`. Only 50 inputs, so a rough reading. Rewrites by their model count as AI.",
             "", "| Texts | n | Mean | Median | Likely AI | Likely human | Under 0.5 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, texts in groups.items():
        scores = [score(text) for text in texts]
        ai = sum(value > thresholds["ai"] for value in scores)
        human = sum(value <= thresholds["review"] for value in scores)
        lines.append(f"| {name} | {len(scores)} | {statistics.mean(scores):.3f} | {statistics.median(scores):.3f} | "
                     f"{ai} | {human} | {sum(value < 0.5 for value in scores)} |")
        progress(lines[-1])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, default=DEMOS)
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/humanize_demo.md")
    parser.add_argument("--limit", type=int, help="this many texts of each kind, for a quick test")
    args = parser.parse_args()
    from eval.bench_math import LOGITS
    from eval.math_attack import CachedScorer
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    groups = {name: texts[:args.limit] for name, texts in demo_groups(args.demos).items()}
    scorer = CachedScorer(detector, path=LOGITS)
    text = report(groups, scorer, detector.thresholds, lambda line: print(line, flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
