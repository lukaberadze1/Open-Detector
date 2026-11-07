"""The overnight scoreboard: every revise setup, by model and by document, since runs of one document aren't
independent. A pass must clear the Likely human line and keep its facts. The independent detector shows what isn't just
tuned to Desklib, before and after. Then what the runs can settle: whether a setup beats the baseline, whether a lower
first draft ends lower, and why revisions get dropped. `--export` writes the baseline's revised texts for the math
script.

    python -m eval.overnight_summary eval/cache/overnight/revise.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean, median

import pandas as pd

from opendetector.combiner import HUMAN_CUTOFF
from opendetector.metrics import wilson_interval

METHODS = ("anchored", "sentences")  # revise_anchored, the page's loop since 6 Nov, or revise_blocks before it
FACTS_KEPT = 70  # percent of names and numbers; a hand-set line, so a pass can't be bought by losing them
VARIANTS = ("+drafts", "+roundtrip", "+clauses", "+freetrip", "+masked", "+anchored", "+human-sources", "+more-docs")


def is_baseline(style: str) -> bool:
    return not any(variant in style for variant in VARIANTS)


def revised(row: dict) -> dict | None:
    """The revision a run made, whichever loop made it."""
    return next((row[method] for method in METHODS if method in row), None)


def final(row: dict) -> float:
    return revised(row)["history"][-1]


def rounds_to_line(row: dict) -> int | None:
    """The revision rounds a run took to reach HUMAN_CUTOFF, 0 for a first draft already there; None if it never did.
    Every round can wear the meaning down, so fewer is better as well as cheaper."""
    return next((number for number, value in enumerate(revised(row)["history"])
                 if value is not None and value <= HUMAN_CUTOFF), None)


def passed(row: dict) -> bool:
    facts = revised(row).get("facts")
    return final(row) <= HUMAN_CUTOFF and (facts is None or facts >= FACTS_KEPT)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided chance of a split at least this lopsided if the setup made no difference."""
    total = wins + losses
    if not total:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(wins, losses) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def doc_means(rows: list[dict]) -> dict[str, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["document"]].append(final(row))
    return {document: mean(values) for document, values in groups.items()}


def paired(base: list[dict], other: list[dict]) -> str:
    """Against the baseline, on the runs both reached, so a quota that ran out midway doesn't bias it."""
    reached = {(row["document"], row["repeat"]) for row in base} & {(row["document"], row["repeat"]) for row in other}
    if not reached:
        return "no runs in common with the baseline"
    a = doc_means([row for row in base if (row["document"], row["repeat"]) in reached])
    b = doc_means([row for row in other if (row["document"], row["repeat"]) in reached])
    changes = [b[document] - a[document] for document in a]
    wins, losses = sum(c < -0.005 for c in changes), sum(c > 0.005 for c in changes)
    p_value = sign_test(wins, losses)
    call = "a real difference" if p_value < 0.05 else "too close to call on this many documents"
    return (f"{100 * mean(changes):+.0f} points on average over {len(changes)} documents; lower on {wins}, higher on "
            f"{losses} (sign test p = {p_value:.2f}: {call})")


def setup_line(model: str, name: str, rows: list[dict], outside: dict[str, float]) -> str:
    ends = [final(row) for row in rows]
    means = doc_means(rows)
    facts = [revised(row)["facts"] for row in rows if revised(row).get("facts") is not None]
    passes = sum(map(passed, rows))
    low, high = wilson_interval(passes, len(rows))
    rounds = [rounds_to_line(row) for row in rows]  # one round is the goal: each one can wear the meaning down
    reached = [number for number in rounds if number is not None]
    one_round = sum(number is not None and number <= 1 for number in rounds) / len(rows)
    before = [outside[row["draft"]] for row in rows if row.get("draft") in outside]
    after = [outside[revised(row)["text"]] for row in rows if revised(row)["text"] in outside]
    requests = [row.get("requests", 0) + row.get("revise_requests", 0) + row.get("attack_requests", 0) for row in rows]
    return (f"| {model} | {name} | {len(rows)} | {len(means)} | {100 * mean(row['start'] for row in rows):.0f} | "
            f"{100 * median(ends):.0f} | {sum(end <= HUMAN_CUTOFF for end in ends)} | "
            f"{passes} ({low:.0%} to {high:.0%}) | {sum(value <= HUMAN_CUTOFF for value in means.values())} | "
            f"{mean(facts) if facts else 0:.0f} | "
            f"{median(reached) if reached else '-'} | {one_round:.0%} | "
            f"{f'{100 * mean(before):.0f} > {100 * mean(after):.0f}' if before and after else '-'} | "
            f"{mean(requests):.0f} | {mean(row.get('seconds', 0) for row in rows) / 60:.1f} |")


def draft_effect(rows: list[dict]) -> list[str]:
    """Whether a lower first draft ends lower, within a document as well as across them: multi-start's whole premise."""
    frame = pd.DataFrame([{"document": row["document"], "start": row["start"], "end": final(row)} for row in rows])
    if len(frame) < 4 or frame.start.nunique() < 2:
        return []
    lines = [f"- First draft against final score, all runs: Spearman {frame.start.corr(frame.end, 'spearman'):.2f}"]
    within = frame.groupby("document")[["start", "end"]].transform(lambda column: column - column.mean())
    if within.start.abs().sum() > 0:
        lines.append(f"- The same, within each document: Spearman {within.start.corr(within.end, 'spearman'):.2f}. "
                     "This is the one that says a better draft helps, and not just that some documents are easy.")
    spread = frame.groupby("document").end.agg(lambda column: column.max() - column.min())
    lines.append(f"- Spread of final scores between repeats of a document: {100 * spread.mean():.0f} points on average")
    return lines


def rejections(rows: list[dict]) -> list[str]:
    """Why rewrites get dropped, by document, hardest first: is the meaning check what stalls the hard documents?"""
    lines = ["| Document | Runs | Start % | End % | Rewrites | Dropped: facts | Dropped: style | Unchanged "
             "| Passages kept | Not lower | Facts failures by kind |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["document"]].append(row)
    for document, group in sorted(groups.items(), key=lambda item: -mean(final(row) for row in item[1])):
        counts = Counter()
        for row in group:
            counts.update(revised(row).get("outcomes", {}))
        kinds = ", ".join(f"{key.split(':')[1]} {value}" for key, value in counts.most_common() if ":" in key)
        total = counts["unchanged"] + counts["facts"] + counts["style"] + counts["scored"] or 1
        lines.append(f"| {document} | {len(group)} | {100 * mean(row['start'] for row in group):.0f} | "
                     f"{100 * mean(final(row) for row in group):.0f} | {total} | {counts['facts'] / total:.0%} | "
                     f"{counts['style'] / total:.0%} | {counts['unchanged'] / total:.0%} | {counts['kept']} | "
                     f"{counts['not_lower']} | {kinds or '-'} |")
    return lines


def math_lines(path: Path) -> list[str]:
    """What the math script added after revision; it saves only the texts it changed."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if "score_after" in row]
    if not rows:
        return []
    before, after = [row["score_before"] for row in rows], [row["score_after"] for row in rows]
    crossed = sum(a > HUMAN_CUTOFF >= b for a, b in zip(before, after, strict=True))
    return ["", "## The math script after revision", "",
            f"{len(rows)} revised texts it changed: mean {100 * mean(before):.0f}% to {100 * mean(after):.0f}%, "
            f"{crossed} crossed the {HUMAN_CUTOFF:.0%} line."]


def scoreboard(rows: list[dict], outside: dict[str, float], math: Path | None = None) -> str:
    setups = defaultdict(list)
    for row in rows:
        setups[row["model"], row["style"]].append(row)
    lines = ["# Overnight scoreboard", "", f"Pass: {HUMAN_CUTOFF:.0%} or under with {FACTS_KEPT}% of names and "
             "numbers kept. Independent is a second detector the runs never saw, first draft > final. Compare setups "
             "only within a model.", "",
             "| Model | Setup | Runs | Docs | Draft % | Median end % | At or under the line | Pass "
             "| Docs under on average | Facts % | Rounds to the line | Under after one round | Independent % "
             "| Requests a run "
             "| Minutes a run |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [setup_line(model, style, group, outside)
              for (model, style), group in sorted(setups.items(), key=lambda item: (item[0][0], len(item[0][1])))]
    for model in sorted({model for model, _ in setups}):
        base = next((group for (m, style), group in setups.items() if m == model and is_baseline(style)), None)
        if base is None:
            lines += ["", f"## {model}", "", "No baseline runs, so no comparison."]
            continue
        lines += ["", f"## {model}", "", "### Against the baseline", ""]
        lines += [f"- `{style}`: {paired(base, group)}" for (m, style), group in setups.items()
                  if m == model and not is_baseline(style)] or ["- Nothing else ran."]
        lines += ["", "### Does the first draft matter?", "", *(draft_effect(base) or ["- Too few runs."])]
        lines += ["", "### Why revisions get dropped (baseline)", "", *rejections(base)]
    lines += math_lines(math) if math else []
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path("eval/cache/overnight/summary.md"))
    parser.add_argument("--math", type=Path, default=Path("eval/cache/overnight/revised_math.jsonl"))
    parser.add_argument("--export", type=Path,
                        help="write the baseline's revised texts as AI-labelled .jsonl, then stop")
    parser.add_argument("--no-independent", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.open(encoding="utf-8")]
    rows = [row for row in rows if revised(row)]
    if not rows:
        raise SystemExit("no finished runs")
    if args.export:
        with args.export.open("w", encoding="utf-8") as sink:
            for row in rows:
                if is_baseline(row["style"]):
                    sink.write(json.dumps({"text": revised(row)["text"], "label": 1, "group_key": row["source"],
                                           "model": row["model"], "attack": row["style"]}) + "\n")
        return
    outside = {}
    if not args.no_independent:
        from eval.bench_math import independent_scores

        texts = sorted({revised(row)["text"] for row in rows} | {row["draft"] for row in rows if row.get("draft")})
        outside = dict(zip(texts, independent_scores(texts), strict=True))
    args.report.write_text(scoreboard(rows, outside, args.math), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
