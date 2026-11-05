"""Shipped combiner against a trial one, on the same texts: the test side of each public cache (by `group_key`, as the
fit splits) and every text of a red-team cache, which neither fit saw. Writes nothing to `models/`.

    python -m eval.trial_compare models/trial_combiner_slim.json eval/cache/daigt.jsonl eval/cache/mage.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from opendetector.combiner import Combiner
from opendetector.config import DetectorConfig
from opendetector.metrics import auroc

from .fit_combiner import assign_split

SHIPPED = Path("models/combiner_slim.json")


def rates(combiner: Combiner, frame: pd.DataFrame) -> str:
    scores = combiner.predict_proba(frame.to_dict("records"))
    ai, human = scores[frame.label.values == 1], scores[frame.label.values == 0]
    cuts = combiner.thresholds
    cells = [f"{(ai > cuts['ai']).mean():.0%} | {(ai <= cuts['review']).mean():.0%}" if len(ai) else "- | -",
             f"{(human > cuts['ai']).mean():.0%} | {(human <= cuts['review']).mean():.0%}" if len(human) else "- | -",
             f"{auroc(frame.label.values, scores):.3f}" if len(ai) and len(human) else "-"]
    return " | ".join(cells)


def compare(shipped: Combiner, trial: Combiner, sets: dict[str, pd.DataFrame]) -> str:
    lines = ["# Trial combiner against the shipped one", "", "Written by `python -m eval.trial_compare`. Red-team "
             "texts "
             "were in neither fit. A better trial catches more AI without flagging more human text.", "",
             "| Texts | AI | Human | Combiner | AI caught | AI called human | Human flagged | Human called human "
             "| AUROC |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name, frame in sets.items():
        counts = f"{int((frame.label == 1).sum())} | {int((frame.label == 0).sum())}"
        lines += [f"| {name} | {counts} | {label} | {rates(combiner, frame)} |"
                  for label, combiner in (("shipped", shipped), ("trial", trial))]
    return "\n".join(lines) + "\n"


def usable(path: Path, min_words: int) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    frame = frame[frame.words >= min_words]
    return frame[frame.prose] if "prose" in frame else frame


def collect_samples(path: Path, limit: int | None = None) -> int:
    """Every saved red-team sample as one file for `eval.featurize local`, with an id per text, since older setups
    share ids and featurizing keeps one row per id. Returns how many."""
    from .humanizer_attack import OUT as HUMANIZERS
    from .reference_benchmark import EXTRA_SAMPLES
    from .redteam_report import SAMPLES, load_samples

    saved = load_samples([*SAMPLES, *EXTRA_SAMPLES, HUMANIZERS])
    rows = {hashlib.sha256(row["text"].encode()).hexdigest()[:24]: row for row in saved}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for key, row in list(rows.items())[:limit]:
            sink.write(json.dumps({**row, "id": key, "label": int(row.get("label", 1))}, ensure_ascii=False) + "\n")
    return min(len(rows), limit) if limit else len(rows)


def by_guidance(name: str, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A red-team cache split by whether the detector's scores guided the writing: the guided part is what a better
    combiner has to catch."""
    from .reference_benchmark import is_guided

    guided = frame.attack.map(is_guided)
    humanizer = frame.attack.str.startswith("humanizer+") & ~guided  # public tools, which never saw these scores
    parts = (("guided", frame[guided]), ("public humanizers", frame[humanizer]),
             ("not guided", frame[~guided & ~humanizer]))
    return {f"{name}, {label}": part for label, part in parts if len(part)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial", type=Path)
    parser.add_argument("public", nargs="+", type=Path, help="caches to compare on their test side")
    parser.add_argument("--redteam", nargs="*", type=Path, default=[], help="caches to compare whole")
    parser.add_argument("--seed", type=int, default=0, help="the seed the trial's fit split with")
    parser.add_argument("--report", type=Path, default=Path("eval/reports/trial_slim.md"))
    args = parser.parse_args()
    min_words = DetectorConfig().min_words
    sets = {}
    for path in args.public:
        frame = usable(path, min_words)
        sets[path.stem] = frame[assign_split(frame.group_key, args.seed) == "test"]
    for path in args.redteam:
        sets.update(by_guidance(path.stem, usable(path, min_words)))
    report = compare(Combiner.load(SHIPPED), Combiner.load(args.trial), sets)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
