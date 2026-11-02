"""Compares first drafts: what each change does to a rewrite's score before any revising. Drafts that start under
80% reached the bar far more often, and a draft alone costs a few requests where a revised one costs dozens, so
this screens every change first. It runs eval.tune_revise --rounds 0 once per setup, each change on its own, then
writes eval/reports/drafts.md, best first. Each setup keeps its runs in eval/cache/drafts/, so a stopped run resumes.
Everything written is logged as labelled AI data. Spends quota: a few requests a draft. Moving clauses around isn't
here: on 20 saved rewrites it left the mean at 72.4% (2026-11-02), since it changes only a sentence or two. Nor is
the free round trip, which Google's bot check refused on its first request the same night.
Screen on documents other than those kept back for the final proof (data/essays/sources/fresh/).

    python -m eval.tune_drafts data/essays/sources/claude_salamis.md data/essays/sources/revise/mage-15379.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from eval.tune_revise import NOTES

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {"base": [], "temperature 1.0": ["--temperature", "1.0"], "temperature 1.3": ["--temperature", "1.3"],
          "top-p 0.8": ["--top-p", "0.8"], "top-k 20": ["--top-k", "20"],
          "round trip": ["--attack", "roundtrip-german"],
          **{note: ["--note", note] for note in NOTES}}


def table(results: dict[str, list[dict]]) -> str:
    """Setups by mean starting score, lowest first, each against the base setup."""
    base = [row["start"] for row in results.get("base", [])]
    lines = ["# First drafts", "", "A rewrite's AI likelihood before revising, each change on its own, lowest first. "
             "Written by `python -m eval.tune_drafts`. With a few runs each, small differences are noise.", "",
             "| Setup | Runs | Mean % | Range % | Under 80% | Against base | Names, numbers kept % | Words kept % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    ranked = sorted(((name, rows) for name, rows in results.items() if rows),
                    key=lambda item: statistics.mean(row["start"] for row in item[1]))
    for name, rows in ranked:
        starts = [row["start"] for row in rows]
        facts = [row["facts"] for row in rows if row.get("facts") is not None]
        kept = [row["words_kept"] for row in rows if row.get("words_kept") is not None]
        against = f"{100 * (statistics.mean(starts) - statistics.mean(base)):+.0f}" if base else "-"
        lines.append(f"| {name} | {len(rows)} | {100 * statistics.mean(starts):.0f} | {100 * min(starts):.0f}-"
                     f"{100 * max(starts):.0f} | {sum(start < 0.8 for start in starts)} | {against} | "
                     + (f"{statistics.mean(facts):.0f}" if facts else "-") + " | "
                     + (f"{statistics.mean(kept):.0f}" if kept else "-") + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="+")
    parser.add_argument("--style", default=str(ROOT / "data/styles/alexandria_profile.json"))
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--setups", help=f"comma-separated, of: {', '.join(SETUPS)}")
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/drafts.md")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.setups.split(",")] if args.setups else list(SETUPS)
    if unknown := set(names) - set(SETUPS):
        parser.error(f"unknown setups: {', '.join(sorted(unknown))}")
    folder = ROOT / "eval/cache/drafts"
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        out = folder / f"{Path(args.style).stem}-{args.model}-{name.replace(' ', '-')}.jsonl"
        print(f"=== {name}", flush=True)
        code = subprocess.run([sys.executable, "-m", "eval.tune_revise", "--as-page", "--rounds", "0",
                               "--style", args.style, "--model", args.model, "--repeats", str(args.repeats),
                               "--out", str(out),
                               "--report", str(folder / "last.md"), *SETUPS[name], *args.documents],
                              cwd=ROOT, check=False).returncode
        results[name] = [json.loads(line) for line in out.open(encoding="utf-8")] if out.exists() else []
        args.report.write_text(table(results), encoding="utf-8")
        if code:
            print(f"Stopped at {name}: run it again when there is quota.", file=sys.stderr)
            break
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
