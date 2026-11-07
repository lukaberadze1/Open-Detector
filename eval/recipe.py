"""The Red team page's recipe: which measured features a document rewrite uses, picked from what the overnight runs
measured, with the same priority points that pick the model settings (eval/generation_settings.py).

A feature is a part of a run's setup label that the page can turn on: holding names and numbers exactly (`masked`)
and a round trip through a language (`roundtrip-<language>`). Each setup the runs measured is compared with the
baseline of the same model, so a feature measured on one model only isn't mistaken for that model's own level:
its measures are the pooled baseline plus its average difference from its model's baseline.

Meaning is a floor, not a priority: a setup that keeps under MEANING_FLOOR percent of the original's names and
numbers is never picked while one that keeps more is measured. Points can't trade the meaning away; a run that
got under the detector by dropping what the text said hasn't passed.

`python -m eval.recipe` writes `models/recipe.json` from the overnight runs, as the overnight runner does after
each night. Pure Python, shipped in the downloadable apps.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "models/recipe.json"
RUNS = (ROOT / "eval/cache/overnight/revise.jsonl", ROOT / "eval/cache/tune_revise.jsonl")
HUMAN_CUTOFF = 0.15  # opendetector.combiner.HUMAN_CUTOFF, kept here so the page needn't import the combiner
MEANING_FLOOR = 70  # percent of the original's names and numbers a picked setup has to keep
MIN_RUNS = 4  # a setup measured on fewer runs than this isn't offered
MEASURES = ("detector", "style", "speed", "quota")
METHODS = ("anchored", "sentences")  # revise_anchored (the page's loop since 6 Nov), then revise_blocks
# Label parts that make a setup, and whether the page can turn each on. Multi-start (drafts) is only in tune_revise.
FEATURES = {"masked": "Hold names and numbers exactly", "roundtrip": "Round trip through {language}",
            "drafts": "Several first drafts, the lowest kept", "clauses": "Move clauses around",
            "anchored": "Revise from the original's wording"}
OFFERED = ("masked", "roundtrip")


def features_of(style: str) -> tuple[str, ...]:
    """The features in a setup label, in label order: `masked`, `roundtrip-german`, `drafts3` and so on."""
    found = []
    for part in style.split("+")[1:]:
        base = part.split("-")[0].rstrip("0123456789")
        if base in FEATURES:
            found.append(part)
    return tuple(found)


def describe(feature: str) -> str:
    base = feature.split("-")[0].rstrip("0123456789")
    language = feature.split("-", 1)[1].capitalize() if "-" in feature else ""
    return FEATURES.get(base, feature).format(language=language)


def _revised(row: dict) -> dict | None:
    return next((row[method] for method in METHODS if isinstance(row.get(method), dict)), None)


def run_measures(row: dict) -> dict[str, float | None] | None:
    """One run's measures: under the line (1 or 0), facts kept, style distance, minutes, requests, rounds taken."""
    revised = _revised(row)
    if not revised or not revised.get("history"):
        return None
    history = [value for value in revised["history"] if value is not None]
    if not history:
        return None
    rounds = next((number for number, value in enumerate(history) if value <= HUMAN_CUTOFF), None)
    return {"under": float(history[-1] <= HUMAN_CUTOFF), "end": history[-1], "facts": revised.get("facts"),
            "distance": revised.get("distance"), "minutes": row.get("seconds", 0) / 60,
            "requests": row.get("requests", 0) + row.get("revise_requests", 0) + row.get("attack_requests", 0),
            "rounds": rounds}


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def build(rows: list[dict]) -> dict:
    """The table: each setup's measures by model, and its pooled estimate against the baseline."""
    by_setup: dict[tuple[str, ...], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "document" not in row or not (measured := run_measures(row)):
            continue
        by_setup[features_of(row["style"])][row["model"]].append(measured)
    names = ("under", "end", "facts", "distance", "minutes", "requests", "rounds")
    baseline = by_setup.get((), {})
    pooled = {name: _mean([_mean([run[name] for run in runs]) for runs in baseline.values()]) for name in names}
    setups = []
    for features, models in by_setup.items():
        estimate, runs = {}, sum(len(group) for group in models.values())
        for name in names:
            if not features:
                estimate[name] = pooled[name]
                continue
            gaps = [(own - base, len(group)) for model, group in models.items()
                    if model in baseline and (own := _mean([run[name] for run in group])) is not None
                    and (base := _mean([run[name] for run in baseline[model]])) is not None]
            if gaps and pooled[name] is not None:
                estimate[name] = pooled[name] + sum(gap * n for gap, n in gaps) / sum(n for _, n in gaps)
                top = {"under": 1.0, "end": 1.0, "facts": 100.0}.get(name)  # an added gap can overshoot a share
                estimate[name] = max(0.0, min(top, estimate[name]) if top is not None else estimate[name])
            else:
                estimate[name] = _mean([run[name] for group in models.values() for run in group])
        setups.append({"features": list(features), "runs": runs, "models": sorted(models),
                       "loop": "anchored" if "anchored" in features else "sentences", **estimate})
    setups.sort(key=lambda setup: (len(setup["features"]), setup["features"]))
    return {"setups": setups, "meaning_floor": MEANING_FLOOR, "min_runs": MIN_RUNS,
            "runs": sum(setup["runs"] for setup in setups)}


def load(path: Path | None = None) -> dict | None:
    path = path or TABLE  # looked up at call time, so tests can point it elsewhere
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return table if isinstance(table, dict) and table.get("setups") else None


@dataclass
class Recipe:
    features: tuple[str, ...]
    measures: dict = field(default_factory=dict)
    below_floor: bool = False  # nothing measured kept enough of the meaning: this kept the most

    def has(self, base: str) -> str | None:
        """The feature of this kind, as named in a label (`roundtrip-german`), or None."""
        return next((feature for feature in self.features if feature.split("-")[0].rstrip("0123456789") == base),
                    None)


def offered(setup: dict, allowed: tuple[str, ...] = OFFERED) -> bool:
    """Whether the page can run a setup: every feature one it can turn on, under the page's revision loop."""
    return all(feature.split("-")[0].rstrip("0123456789") in (*allowed, "anchored")
               for feature in setup["features"])


def choose(table: dict, points: dict[str, int], allowed: tuple[str, ...] = OFFERED) -> Recipe | None:
    """The setup these points favour among those measured on MIN_RUNS runs or more and keeping the meaning floor.
    Each measure is scaled worst to best over those setups: getting under the line (detector), the style profile's
    distance (style), minutes (speed) and requests (quota). Ties go to the one keeping more names and numbers."""
    floor = table.get("meaning_floor", MEANING_FLOOR)
    pool = [setup for setup in table.get("setups", [])
            if setup.get("runs", 0) >= table.get("min_runs", MIN_RUNS) and offered(setup, allowed)]
    if not pool:
        return None
    keeping = [setup for setup in pool if (setup.get("facts") or 0) >= floor]
    below = not keeping
    candidates = keeping or [max(pool, key=lambda setup: setup.get("facts") or 0)]

    def scaled(name: str, higher_better: bool) -> list[float]:
        values = [setup.get(name) for setup in candidates]
        known = [value for value in values if value is not None]
        if not known or max(known) == min(known):
            return [0.5] * len(values)
        low, high = min(known), max(known)
        return [0.5 if value is None else ((value - low) if higher_better else (high - value)) / (high - low)
                for value in values]

    columns = {"detector": scaled("under", True), "style": scaled("distance", False),
               "speed": scaled("minutes", False), "quota": scaled("requests", False)}
    spent = sum(points.get(measure, 0) for measure in MEASURES)
    weights = {measure: points.get(measure, 0) / spent if spent else 1 / len(MEASURES) for measure in MEASURES}

    def value(index: int) -> tuple[float, float]:
        return (sum(weights[m] * columns[m][index] for m in MEASURES), candidates[index].get("facts") or 0)

    best = candidates[max(range(len(candidates)), key=value)]
    features = tuple(feature for feature in best["features"] if feature != "anchored")
    return Recipe(features, {name: best.get(name) for name in
                             ("runs", "under", "end", "facts", "distance", "minutes", "requests", "rounds", "loop")},
                  below)


def report(table: dict) -> str:
    lines = [f"# Recipe table, from {table['runs']} runs", "",
             f"Under the line is the share of runs at or under {HUMAN_CUTOFF:.0%}. Each setup is its model's baseline "
             f"plus its average difference from it. The page picks only setups keeping {table['meaning_floor']}% of "
             f"names and numbers, measured on {table['min_runs']} runs or more.", "",
             "| Setup | Loop | Runs | Under the line | Mean end | Facts % | Rounds to the line | Minutes | Requests |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def cell(value, form):
        return "-" if value is None else format(value, form)

    for setup in table["setups"]:
        name = ", ".join(describe(f) for f in setup["features"] if f != "anchored") or "Baseline"
        lines.append(f"| {name} | {setup['loop']} | {setup['runs']} | {cell(setup['under'], '.0%')} | "
                     f"{cell(setup['end'], '.0%')} | {cell(setup['facts'], '.0f')} | {cell(setup['rounds'], '.1f')} | "
                     f"{cell(setup['minutes'], '.1f')} | {cell(setup['requests'], '.0f')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path,
                        help=f"tune_revise outputs (default: {', '.join(map(str, RUNS))})")
    parser.add_argument("--out", type=Path, default=TABLE)
    parser.add_argument("--report", type=Path, help="also write the table as Markdown")
    args = parser.parse_args(argv)
    rows = []
    for path in args.runs or RUNS:
        if path.exists():
            for line in path.open(encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a line half-written by a run still going
    table = build(rows)
    if not table["setups"]:
        print("No revised runs to build from.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    text = report(table)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
