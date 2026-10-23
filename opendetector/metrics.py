"""Detection metrics. Detectors are judged by how much AI text they catch at a fixed,
low false-positive rate, not by accuracy: one wrongly accused student matters more than
a few missed essays."""

from __future__ import annotations

import math

import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability a random AI text outscores a random human one (ties count half)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    ranks = np.empty(len(values))
    ranks[values.argsort(kind="mergesort")] = np.arange(1, len(values) + 1)
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]  # average ranks over ties
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> tuple[float, float]:
    """Share of AI texts caught when at most `fpr` of human texts score above the threshold.
    Returns (tpr, threshold); texts are flagged when score > threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    human = np.sort(scores[labels == 0])[::-1]
    ai = scores[labels == 1]
    if len(human) == 0 or len(ai) == 0:
        return float("nan"), float("nan")
    k = math.floor(fpr * len(human))
    threshold = human[k] if k < len(human) else -math.inf
    return float((ai > threshold).mean()), float(threshold)


def conformal_threshold(human_scores: np.ndarray, alpha: float) -> float:
    """Threshold t such that a new human text has P(score > t) <= alpha, with a
    finite-sample guarantee when calibration and deployment texts are exchangeable."""
    s = np.sort(np.asarray(human_scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if 0 < k <= len(s) else 1.0


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)
