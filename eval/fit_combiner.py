"""Fit the combiner on cached features, set thresholds, and write an evaluation report.

    python -m eval.fit_combiner eval/cache/*.jsonl

Rows are split by `group_key` into development (80%) and test (20%). Thresholds are set by
cross-fitting: the development set is cut into 5 folds, each fold is scored by a model
fitted on the other four, and the flag thresholds come from those out-of-fold scores on
every development human text. The final model is then fitted on all of development, and
every number in the report comes from the untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from opendetector.combiner import DEFAULT_ALPHAS, fit_logistic, set_thresholds
from opendetector.config import DetectorConfig
from opendetector.metrics import auroc, tpr_at_fpr, wilson_interval

from .datasets import COLUMNS

META = set(COLUMNS) | {"words"}
# Each engine's headline feature, oriented so that higher = more AI-like.
SINGLE_ENGINE = {"GLYPH": ("glyph_mean", 1), "Desklib": ("desklib_mean", 1), "Binoculars": ("bino_doc", -1)}
FOLDS = 5


def _hash(seed: int, salt: str, key: str) -> int:
    return int(hashlib.sha1(f"{seed}:{salt}{key}".encode()).hexdigest()[:8], 16)


def assign_split(keys: pd.Series, seed: int) -> pd.Series:
    return keys.map(lambda k: "dev" if _hash(seed, "", k) / 0x100000000 < 0.8 else "test")


def out_of_fold_scores(dev: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    """Score each development row with a model that never saw its source document."""
    folds = dev.group_key.map(lambda k: _hash(seed, "fold:", k) % FOLDS).values
    scores = np.empty(len(dev))
    for k in range(FOLDS):
        held = folds == k
        model = fit_logistic(dev[features].values[~held], dev.label.values[~held], features)
        scores[held] = model.predict_matrix(dev[features].values[held])
    return scores


def source_level(sources: np.ndarray, flags: np.ndarray) -> tuple[int, int]:
    """(source documents with any flagged version, source documents)."""
    per_source = pd.Series(flags).groupby(sources).max()
    return int(per_source.sum()), len(per_source)


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
    # RAID's "paraphrase" attack rewrites text with the DIPPER language model, so its "human"
    # rows are machine-paraphrased; scoring them as human would count correct catches as errors.
    ai_rewritten = (df.dataset == "raid") & (df.label == 0) & (df.attack == "paraphrase")
    df = df[~short & ~ai_rewritten].reset_index(drop=True)
    features = [c for c in df.columns if c not in META and not df[c].isna().all()]
    df["split"] = assign_split(df.group_key, args.seed)
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    n_dev_human = (dev.label == 0).sum()
    needed = round(1 / DEFAULT_ALPHAS["ai"])
    if n_dev_human < needed:
        raise SystemExit(f"Only {n_dev_human} human texts in the development split; "
                         f"a {DEFAULT_ALPHAS['ai']:.0%} FPR threshold needs at least {needed}.")

    oof = out_of_fold_scores(dev, features, args.seed)
    combiner = fit_logistic(
        dev[features].values, dev.label.values, features,
        info={"n_fit": len(dev), "datasets": sorted(df.dataset.unique()), "fitted": date.today().isoformat(),
              "threshold_method": f"{FOLDS}-fold out-of-fold conformal"},
    )
    set_thresholds(combiner, oof[dev.label.values == 0])

    y = test.label.values
    p = combiner.predict_matrix(test[features].values)
    scores = {name: sign * test[col].values for name, (col, sign) in SINGLE_ENGINE.items() if col in test}
    scores["Combined"] = p
    flagged = p > combiner.thresholds["ai"]
    review = p > combiner.thresholds["review"]
    human, ai = y == 0, y == 1
    fp, nh = int(flagged[human].sum()), int(human.sum())
    # Attacked copies of one document are not independent errors, so the confidence
    # intervals count source documents, flagged if any version of them is.
    sources = test[human].group_key.values
    src_fp, n_src = source_level(sources, flagged[human])
    src_lo, src_hi = wilson_interval(src_fp, n_src)
    rev_fp, _ = source_level(sources, review[human])
    rev_lo, rev_hi = wilson_interval(rev_fp, n_src)

    combiner.info.update({
        "n_test": len(test),
        "test_auroc": auroc(y, p),
        "test_tpr_at_1pct_fpr": tpr_at_fpr(y, p, 0.01)[0],
        "test_fpr_at_ai_threshold": fp / nh if nh else None,
        "test_source_fpr_at_ai_threshold": src_fp / n_src if n_src else None,
        "test_tpr_at_ai_threshold": float(flagged[ai].mean()) if ai.any() else None,
    })
    combiner.save(args.out)

    def verdict_on_target(observed: float, alpha: float, lower: float) -> str:
        if lower > alpha:
            return " **Above target even at the low end of the interval: get more human calibration data.**"
        return " (above target, but within sampling error)" if observed > alpha else ""

    weights = sorted(zip(features, combiner.coef, strict=True), key=lambda t: -abs(t[1]))
    report = [
        f"# Evaluation report ({date.today().isoformat()})", "",
        f"Data: {', '.join(args.caches)}. Dropped {int(short.sum())} texts under {args.min_words} words and "
        f"{int(ai_rewritten.sum())} RAID \"human\" texts rewritten by its AI paraphraser.",
        f"Split by source document: development {len(dev)} ({n_dev_human} human, thresholds set on "
        f"{FOLDS}-fold out-of-fold scores), test {len(test)} ({nh} human texts from {n_src} source documents, "
        f"{int(ai.sum())} AI).", "",
        "## Detectors on the test split", "", *score_table(y, scores), "",
        "## Deployed thresholds", "",
        "False-positive rates count human source documents (flagged if any version is); per-text rates follow.", "",
        f"- **Likely AI** when p > {combiner.thresholds['ai']:.4f} (target FPR {DEFAULT_ALPHAS['ai']:.0%}): "
        f"flags {src_fp}/{n_src} human documents, {src_fp / n_src:.2%} (95% CI {src_lo:.2%}–{src_hi:.2%}); "
        f"per text {fp}/{nh}, {fp / nh:.2%}. Catches {flagged[ai].mean():.1%} of AI texts."
        + verdict_on_target(src_fp / n_src, DEFAULT_ALPHAS["ai"], src_lo),
        f"- **Needs review** when p > {combiner.thresholds['review']:.4f} (target FPR {DEFAULT_ALPHAS['review']:.0%}): "
        f"flags {rev_fp}/{n_src} human documents, {rev_fp / n_src:.2%} (95% CI {rev_lo:.2%}–{rev_hi:.2%}). "
        f"Catches {review[ai].mean():.1%} of AI texts."
        + verdict_on_target(rev_fp / n_src, DEFAULT_ALPHAS["review"], rev_lo), "",
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
