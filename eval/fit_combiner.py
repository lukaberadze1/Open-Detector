"""Fit the combiner on cached features, set thresholds, and write an evaluation report.

    python -m eval.fit_combiner eval/cache/*.jsonl

Rows are split by `group_key` into fit (60%), calibration (20%) and test (20%). The
logistic regression is fitted on the first, flag thresholds are set on the human texts of
the second, and every number in the report comes from the third.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from opendetector.combiner import DEFAULT_ALPHAS, fit_combiner
from opendetector.config import DetectorConfig
from opendetector.metrics import auroc, tpr_at_fpr, wilson_interval

from .datasets import COLUMNS

META = set(COLUMNS) | {"words"}
# Each engine's headline feature, oriented so that higher = more AI-like.
SINGLE_ENGINE = {"GLYPH": ("glyph_mean", 1), "Desklib": ("desklib_mean", 1), "Binoculars": ("bino_doc", -1)}


def assign_split(keys: pd.Series, seed: int) -> pd.Series:
    def bucket(key: str) -> str:
        u = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0x100000000
        return "fit" if u < 0.6 else "cal" if u < 0.8 else "test"

    return keys.map(bucket)


def score_table(y: np.ndarray, scores: dict[str, np.ndarray]) -> list[str]:
    lines = ["| Detector | AUROC | TPR @ 1% FPR | TPR @ 5% FPR |", "|---|---|---|---|"]
    for name, s in scores.items():
        ok = ~np.isnan(s)
        lines.append(f"| {name} | {auroc(y[ok], s[ok]):.4f} | {tpr_at_fpr(y[ok], s[ok], 0.01)[0]:.1%} "
                     f"| {tpr_at_fpr(y[ok], s[ok], 0.05)[0]:.1%} |")
    return lines


def slice_table(test: pd.DataFrame, flagged: np.ndarray, column: str) -> list[str]:
    lines = [f"| {column} | human n | FPR (95% CI) | AI n | TPR |", "|---|---|---|---|---|"]
    for value, part in test.groupby(column):
        f = flagged[part.index]
        human, ai = part.label.values == 0, part.label.values == 1
        fp, nh = int(f[human].sum()), int(human.sum())
        lo, hi = wilson_interval(fp, nh)
        fpr = f"{fp / nh:.2%} ({lo:.2%}–{hi:.2%})" if nh else "–"
        tpr = f"{f[ai].mean():.1%}" if ai.any() else "–"
        lines.append(f"| {value} | {nh} | {fpr} | {int(ai.sum())} | {tpr} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("caches", nargs="+", help="JSONL files written by eval.featurize")
    parser.add_argument("--out", default=str(DetectorConfig().combiner_path))
    parser.add_argument("--report", default="eval/reports/latest.md")
    parser.add_argument("--min-words", type=int, default=DetectorConfig().min_words,
                        help="drop shorter texts; the deployed detector abstains on them")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.concat([pd.read_json(p, lines=True, dtype={"id": str, "group_key": str}) for p in args.caches],
                   ignore_index=True)
    short = df.words < args.min_words
    df = df[~short].reset_index(drop=True)
    features = [c for c in df.columns if c not in META and not df[c].isna().all()]
    df["split"] = assign_split(df.group_key, args.seed)
    fit, cal = df[df.split == "fit"], df[df.split == "cal"]
    test = df[df.split == "test"].reset_index(drop=True)

    if (cal.label == 0).sum() < 1 / DEFAULT_ALPHAS["ai"]:
        raise SystemExit(f"Only {(cal.label == 0).sum()} human texts in the calibration split; "
                         f"a {DEFAULT_ALPHAS['ai']:.0%} FPR threshold needs at least {int(1 / DEFAULT_ALPHAS['ai'])}.")

    combiner = fit_combiner(
        fit[features].values, fit.label.values, features, cal[features].values, cal.label.values,
        info={"n_fit": int(len(fit)), "datasets": sorted(df.dataset.unique()), "fitted": date.today().isoformat()},
    )

    y = test.label.values
    p = combiner.predict_matrix(test[features].values)
    scores = {name: sign * test[col].values for name, (col, sign) in SINGLE_ENGINE.items() if col in test}
    scores["Combined"] = p
    flagged = p > combiner.thresholds["ai"]
    review = p > combiner.thresholds["review"]
    human, ai = y == 0, y == 1
    fp, nh = int(flagged[human].sum()), int(human.sum())
    lo, hi = wilson_interval(fp, nh)

    combiner.info.update({
        "n_test": int(len(test)),
        "test_auroc": auroc(y, p),
        "test_tpr_at_1pct_fpr": tpr_at_fpr(y, p, 0.01)[0],
        "test_fpr_at_ai_threshold": fp / nh if nh else None,
        "test_tpr_at_ai_threshold": float(flagged[ai].mean()) if ai.any() else None,
    })
    combiner.save(args.out)

    weights = sorted(zip(features, combiner.coef), key=lambda t: -abs(t[1]))
    report = [
        f"# Evaluation report ({date.today().isoformat()})", "",
        f"Data: {', '.join(args.caches)}. Dropped {int(short.sum())} texts under {args.min_words} words.",
        f"Split by source document: fit {len(fit)}, calibration {len(cal)}, test {len(test)} "
        f"({nh} human, {int(ai.sum())} AI).", "",
        "## Detectors on the test split", "", *score_table(y, scores), "",
        "## Deployed thresholds", "",
        f"- **Likely AI** when p > {combiner.thresholds['ai']:.4f} (target FPR {DEFAULT_ALPHAS['ai']:.0%}): "
        f"test FPR {fp / nh:.2%} (95% CI {lo:.2%}–{hi:.2%}), catches {flagged[ai].mean():.1%} of AI texts.",
        f"- **Needs review** when p > {combiner.thresholds['review']:.4f} (target FPR {DEFAULT_ALPHAS['review']:.0%}): "
        f"test FPR {review[human].mean():.2%}, catches {review[ai].mean():.1%} of AI texts.", "",
        "## Combiner weights (standardized features)", "",
        *[f"- `{f}`: {w:+.3f}" for f, w in weights], "",
    ]
    for column in ("dataset", "domain", "attack", "model", "group"):
        if test[column].nunique() > 1:
            report += [f"## By {column} (at the Likely-AI threshold)", "", *slice_table(test, flagged, column), ""]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nWrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
