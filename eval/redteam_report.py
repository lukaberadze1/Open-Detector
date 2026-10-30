"""Score the saved red-team samples with the detector: how much of the AI text built to slip past it does.

    python -m eval.redteam_report                    # every saved sample, with the slim build
    python -m eval.redteam_report --profile full     # with every engine (about 3.5 GB)

The Red team page and the settings benchmark save every sample they write as labelled AI data
(data/essays/local_redteam.jsonl and data/essays/tuning_redteam.jsonl), and, when the user says a source
document is their own writing, its originals as human data beside them. This scores the samples the
detector reads (150 words or more of prose) with the shipped combiner and its thresholds and reports:

- for AI samples, the share it calls Likely AI (caught) and Likely human (missed)
- for human ones, the share it calls Likely AI (wrongly flagged)

each with a Wilson 95% interval, overall and by attack, model, thinking level, temperature, blocks per
request and dataset. The miss rate is the number to drive down when the detector is retrained on these
samples (Phase 2); refitting the combiner alone can only reweight the engines it already has.

Scores are cached in eval/cache/redteam_scores.jsonl by text, profile and combiner, so a second run
scores only what is new, and a refitted combiner scores everything again. Nothing else changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval.redteam_batch import copied_row
from opendetector import DetectorConfig
from opendetector.config import DATA_ROOT
from opendetector.metrics import wilson_interval
from opendetector.segment import count_words, looks_like_prose

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (DATA_ROOT / "data/essays/local_redteam.jsonl", DATA_ROOT / "data/essays/tuning_redteam.jsonl")
CACHE = ROOT / "eval/cache/redteam_scores.jsonl"
BREAKDOWNS = ("attack", "model", "thinking", "temperature", "blocks_per_request", "dataset")


def load_samples(paths: list[Path]) -> list[dict]:
    """Every saved sample the detector reads: 150 words or more of prose, once each by id. An AI-labelled
    rewrite that hands back its original (redteam_batch.copied_row) is the writer's text and left out."""
    rows = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = row.get("text") or ""
            if copied_row(row):
                continue
            if count_words(text) >= DetectorConfig().min_words and looks_like_prose(text):
                rows.setdefault(row.get("id") or hashlib.sha256(text.encode()).hexdigest(), row)
    return list(rows.values())


def score_key(text: str, profile: str, combiner: str) -> str:
    return hashlib.sha256(f"{profile}\n{combiner}\n{text}".encode()).hexdigest()[:24]


def score(rows: list[dict], profile: str, cache: Path = CACHE) -> list[dict]:
    """Each row with the detector's verdict and AI likelihood, from the cache when it has them."""
    config = DetectorConfig.for_profile(profile)
    path = config.combiner_path
    combiner = hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""
    known = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            known[entry["key"]] = entry
    missing = [row for row in rows if score_key(row["text"], profile, combiner) not in known]
    if missing:
        from opendetector import Detector

        print(f"Scoring {len(missing)} of {len(rows)} samples with the {profile} detector…")
        detector = Detector.from_config(config)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as output:
            for row in missing:
                report = detector.analyze(row["text"], sentences=False)
                entry = {"key": score_key(row["text"], profile, combiner), "verdict": report.verdict,
                         "ai_probability": report.ai_probability}
                known[entry["key"]] = entry
                output.write(json.dumps(entry) + "\n")
    return [{**row, **{k: v for k, v in known[score_key(row["text"], profile, combiner)].items() if k != "key"}}
            for row in rows]


def _rate(rows: list[dict], verdict: str) -> str:
    hits = sum(row["verdict"] == verdict for row in rows)
    low, high = wilson_interval(hits, len(rows))
    return f"{hits / len(rows):.0%} ({low:.0%} to {high:.0%})"


def _mean_likelihood(rows: list[dict]) -> str:
    values = [row["ai_probability"] for row in rows if row.get("ai_probability") is not None]
    return f"{statistics.fmean(values):.0%}" if values else "n/a"


def report(rows: list[dict], profile: str) -> str:
    """The scored samples as Markdown."""
    ai = [row for row in rows if row.get("label", 1) == 1 and row.get("verdict")]
    human = [row for row in rows if row.get("label") == 0 and row.get("verdict")]
    lines = ["# Red-team samples against the detector", "",
             f"{len(ai)} AI samples and {len(human)} human ones of 150 words or more, scored with the {profile} "
             "detector and its shipped thresholds. Written by `python -m eval.redteam_report`. Brackets are "
             "Wilson 95% intervals.", ""]
    if ai:
        lines += ["## AI samples", "",
                  f"- Caught (Likely AI): {_rate(ai, 'likely_ai')}",
                  f"- Needs review: {_rate(ai, 'needs_review')}",
                  f"- **Missed (Likely human): {_rate(ai, 'likely_human')}**",
                  f"- Mean AI likelihood: {_mean_likelihood(ai)}", ""]
        for field in BREAKDOWNS:
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in ai:
                value = row.get(field)
                groups["not recorded" if value is None else str(value)].append(row)
            if len(groups) < 2:
                continue
            lines += [f"### By {field.replace('_', ' ')}", "",
                      f"| {field.replace('_', ' ').capitalize()} | Samples | Missed | Caught | Mean AI likelihood |",
                      "|---|---:|---:|---:|---:|"]
            for name, group in sorted(groups.items(), key=lambda pair: -len(pair[1])):
                lines.append(f"| {name} | {len(group)} | {_rate(group, 'likely_human')} | "
                             f"{_rate(group, 'likely_ai')} | {_mean_likelihood(group)} |")
            lines.append("")
    if human:
        lines += ["## Human originals", "",
                  f"- Wrongly flagged (Likely AI): {_rate(human, 'likely_ai')}",
                  f"- Needs review: {_rate(human, 'needs_review')}",
                  f"- Mean AI likelihood: {_mean_likelihood(human)}", ""]
    if not ai and not human:
        lines.append("No saved sample is long enough for the detector yet.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="*", type=Path, help="JSONL files of samples (default: the saved ones)")
    parser.add_argument("--profile", choices=("full", "slim"), default="slim", help="the detector build")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/redteam.md")
    args = parser.parse_args(argv)
    rows = load_samples(args.samples or list(SAMPLES))
    if not rows:
        print("No saved sample is long enough for the detector yet.")
        return 1
    scored = score(rows, args.profile, args.cache)
    text = report(scored, args.profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    ai = [row for row in scored if row.get("label", 1) == 1]
    missed = sum(row["verdict"] == "likely_human" for row in ai)
    print(f"Wrote {args.report}: the detector misses {missed} of {len(ai)} AI samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
