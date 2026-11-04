"""Evaluate ZipPy readings across source-grouped MAGE and RAID samples.

Examples:
    .venv/bin/python -m eval.zippy_cross_source --datasets mage
    .venv/bin/python -m eval.zippy_cross_source --raid-csv /path/to/raid.csv

The report is exploratory. It fits only the local evaluation variants and never updates
the production combiner. MAGE has no document identifier, so exact normalized text hashes
serve as its source groups; RAID uses the source groups supplied by its dataset loader.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from opendetector.combiner import fit_logistic
from opendetector.segment import burstiness, split_sentences
from opendetector.zippy import compression_readings

from .datasets import load_mage, load_raid

ZIPPY_FEATURES = ["zippy_lzma", "zippy_zlib"]
DETERMINISTIC_FEATURES = ["burst_cv", "burst_adj", "bigram_repetition"]
ALL_FEATURES = ZIPPY_FEATURES + DETERMINISTIC_FEATURES
MIN_CALIBRATION_HUMAN = 100  # permits a nontrivial empirical 1% FPR operating point
_WORD = re.compile(r"[\w']+", re.UNICODE)


def _mage_group(text: str) -> str:
    """MAGE lacks document IDs; group exact normalized duplicates together."""
    normalized = " ".join(text.split()).casefold()
    return "mage:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_for_group(group_key: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{group_key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.6:
        return "train"
    if value < 0.8:
        return "calibration"
    return "test"


def _bigram_repetition(text: str) -> float:
    words = _WORD.findall(text.casefold())
    bigrams = list(pairwise(words))
    if not bigrams:
        return math.nan
    return 1.0 - len(set(bigrams)) / len(bigrams)


def _feature_row(text: str) -> dict[str, float]:
    values = compression_readings(text)
    values.update(burstiness(text, split_sentences(text)))
    values["bigram_repetition"] = _bigram_repetition(text)
    return values


def _load_rows(datasets: list[str], mage_n: int, mage_split: str, raid_chunks: int,
               raid_split: str, raid_csv: str | None, seed: int) -> pd.DataFrame:
    frames = []
    if "mage" in datasets:
        mage = load_mage(split=mage_split, n=mage_n, seed=seed)
        mage["group_key"] = mage.text.map(_mage_group)
        frames.append(mage)
    if "raid" in datasets:
        frames.append(load_raid(split=raid_split, chunks=raid_chunks, csv_path=raid_csv, seed=seed))
    if not frames:
        raise ValueError("Choose at least one dataset: mage or raid.")
    rows = pd.concat(frames, ignore_index=True)
    rows["text"] = rows.text.fillna("").astype(str)
    rows = rows[rows.text.str.split().str.len() >= 150].copy()
    rows["group_key"] = rows.apply(lambda row: f"{row.dataset}:{row.group_key}", axis=1)
    rows["split"] = rows.group_key.map(lambda key: _split_for_group(key, seed))
    return rows.reset_index(drop=True)


def _bootstrap_intervals(test: pd.DataFrame, scores: np.ndarray, threshold: float,
                         iterations: int, seed: int) -> dict[str, tuple[float, float]]:
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(test.group_key):
        groups.setdefault(group, []).append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    aucs, accuracies = [], []
    labels = test.label.to_numpy(dtype=int)
    flags = scores > threshold
    for _ in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.fromiter((i for key in sampled for i in groups[key]), dtype=int)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        aucs.append(roc_auc_score(sampled_labels, scores[indices]))
        accuracies.append(balanced_accuracy_score(sampled_labels, flags[indices]))
    if not aucs:
        return {"auroc": (math.nan, math.nan), "balanced_accuracy": (math.nan, math.nan)}
    return {
        "auroc": tuple(np.percentile(aucs, [2.5, 97.5]).tolist()),
        "balanced_accuracy": tuple(np.percentile(accuracies, [2.5, 97.5]).tolist()),
    }


def _threshold_at_fpr(human_scores: np.ndarray, fpr: float = 0.01) -> float:
    ordered = np.sort(np.asarray(human_scores, dtype=float))[::-1]
    if not len(ordered):
        raise ValueError("The calibration split has no human texts.")
    allowed_false_positives = math.floor(fpr * len(ordered))
    return float(ordered[allowed_false_positives])


def _evaluate(rows: pd.DataFrame, features: list[str], bootstrap: int, seed: int) -> dict:
    train = rows[rows.split == "train"].reset_index(drop=True)
    calibration = rows[rows.split == "calibration"].reset_index(drop=True)
    test = rows[rows.split == "test"].reset_index(drop=True)
    if set(train.label) != {0, 1} or set(calibration.label) != {0, 1} or set(test.label) != {0, 1}:
        raise ValueError("A split is missing human or AI examples; increase sample size or add a dataset.")

    model = fit_logistic(train[features].to_numpy(float), train.label.to_numpy(int), features)
    calibration_scores = model.predict_matrix(calibration[features].to_numpy(float))
    human_scores = calibration_scores[calibration.label.to_numpy(int) == 0]
    if len(human_scores) < MIN_CALIBRATION_HUMAN:
        raise ValueError(f"Only {len(human_scores)} human calibration documents; at least "
                         f"{MIN_CALIBRATION_HUMAN} are needed for a useful 1% FPR threshold.")
    threshold = _threshold_at_fpr(human_scores)

    scores = model.predict_matrix(test[features].to_numpy(float))
    labels = test.label.to_numpy(int)
    flags = scores > threshold
    intervals = _bootstrap_intervals(test, scores, threshold, bootstrap, seed)
    domain_rows = []
    for (dataset, domain), part in test.groupby(["dataset", "domain"], sort=True):
        values = scores[part.index.to_numpy()]
        y = part.label.to_numpy(int)
        predicted = values > threshold
        human, ai = y == 0, y == 1
        domain_rows.append({
            "dataset": dataset, "domain": domain,
            "human": int(human.sum()), "ai": int(ai.sum()),
            "fpr": float(predicted[human].mean()) if human.any() else math.nan,
            "tpr": float(predicted[ai].mean()) if ai.any() else math.nan,
        })
    calibration_fpr = float((human_scores > threshold).mean())
    return {
        "threshold": threshold,
        "calibration_fpr": calibration_fpr,
        "test_auroc": float(roc_auc_score(labels, scores)),
        "test_tpr": float(flags[labels == 1].mean()),
        "test_fpr": float(flags[labels == 0].mean()),
        "test_balanced_accuracy": float(balanced_accuracy_score(labels, flags)),
        "intervals": intervals,
        "domain_rows": domain_rows,
        "n_train": len(train), "n_calibration": len(calibration), "n_test": len(test),
        "calibration_human": len(human_scores),
        "test_groups": test.group_key.nunique(),
    }


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def _report(rows: pd.DataFrame, results: dict[str, dict], bootstrap: int, seed: int) -> str:
    lines = [
        f"# Cross-source ZipPy evaluation ({date.today().isoformat()})", "",
        "Exploratory document-grouped evaluation. It does not change the production combiner.",
        "MAGE provides no document IDs; identical normalized texts are grouped by hash. "
        "RAID uses `group_key` from its loader.",
        f"Rows after the 150-word minimum: {len(rows)}; bootstrap replicates: {bootstrap}; seed: {seed}.", "",
        "A logistic model is fitted on training groups, calibrated on separate human groups "
        "to an empirical 1% FPR threshold,",
        "then evaluated once on test groups. Confidence intervals use document-group bootstrap resampling.", "",
        "| Variant | Threshold | Calibration FPR | Test FPR | Test TPR | AUROC (95% CI) | Balanced accuracy (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        ci = result["intervals"]
        lines.append(
            f"| {name} | {result['threshold']:.6f} | {_percent(result['calibration_fpr'])} | "
            f"{_percent(result['test_fpr'])} | {_percent(result['test_tpr'])} | "
            f"{result['test_auroc']:.3f} ({ci['auroc'][0]:.3f} to {ci['auroc'][1]:.3f}) | "
            f"{result['test_balanced_accuracy']:.3f} "
            f"({ci['balanced_accuracy'][0]:.3f} to {ci['balanced_accuracy'][1]:.3f}) |"
        )
    lines.extend([
        "", "## Split sizes", "",
        "| Split | Documents | Source groups | Human | AI |", "|---|---:|---:|---:|---:|",
    ])
    for split in ("train", "calibration", "test"):
        part = rows[rows.split == split]
        lines.append(f"| {split} | {len(part)} | {part.group_key.nunique()} | "
                     f"{int((part.label == 0).sum())} | {int((part.label == 1).sum())} |")
    lines.extend(["", "## Test results by dataset and domain", "",
                  "| Dataset | Domain | Human | AI | FPR | TPR |", "|---|---|---:|---:|---:|---:|"])
    for name, result in results.items():
        lines.extend(["", f"### {name}", ""])
        for row in result["domain_rows"]:
            lines.append(f"| {row['dataset']} | {row['domain']} | {row['human']} | {row['ai']} | "
                         f"{_percent(row['fpr'])} | {_percent(row['tpr'])} |")
    lines.extend([
        "", "## Features and limits", "",
        "ZipPy uses `zippy_lzma` and `zippy_zlib`. The combined variant adds sentence-length variation "
        "(`burst_cv`, `burst_adj`) and a deterministic repeated word-bigram rate.",
        "The 1% threshold is empirical; with fewer than 100 human calibration groups the script stops. "
        "The CI describes test-group resampling, not uncertainty from new domains or data collection.",
        "No result here is an entropy measurement or a production calibration.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="mage,raid", help="comma-separated: mage,raid")
    parser.add_argument("--mage-split", default="test")
    parser.add_argument("--mage-n", type=int, default=4000)
    parser.add_argument("--raid-split", default="train")
    parser.add_argument("--raid-chunks", type=int, default=20)
    parser.add_argument("--raid-csv", help="optional local RAID CSV; otherwise samples byte ranges online")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval/reports/zippy_cross_source.md")
    args = parser.parse_args()
    datasets = [value.strip().lower() for value in args.datasets.split(",") if value.strip()]
    unknown = set(datasets) - {"mage", "raid"}
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")

    rows = _load_rows(datasets, args.mage_n, args.mage_split, args.raid_chunks,
                      args.raid_split, args.raid_csv, args.seed)
    feature_values = [_feature_row(text) for text in rows.text]
    rows = pd.concat([rows.reset_index(drop=True), pd.DataFrame(feature_values)], axis=1)
    results = {
        "ZipPy": _evaluate(rows, ZIPPY_FEATURES, args.bootstrap, args.seed),
        "ZipPy + deterministic features": _evaluate(rows, ALL_FEATURES, args.bootstrap, args.seed),
    }
    report = _report(rows, results, args.bootstrap, args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
