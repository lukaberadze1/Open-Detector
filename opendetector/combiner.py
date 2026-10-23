"""Learned combiner: logistic regression over engine features, plus thresholds chosen on
held-out human text for a target false-positive rate.

This replaces hand-tuned score offsets. Prediction is plain numpy and the model is stored
as JSON, so the app needs neither scikit-learn nor pickle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .metrics import conformal_threshold

# Verdict bands: "ai" is the flag threshold, "review" a looser band for a human look.
DEFAULT_ALPHAS = {"ai": 0.01, "review": 0.05}


@dataclass
class Combiner:
    features: list[str]
    medians: list[float]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    thresholds: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), np.asarray(self.medians), X)
        z = ((X - np.asarray(self.mean)) / np.asarray(self.scale)) @ np.asarray(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        X = np.array([[row.get(f, np.nan) for f in self.features] for row in rows], dtype=float)
        return self.predict_matrix(X.reshape(len(rows), len(self.features)))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> Combiner:
        return cls(**json.loads(Path(path).read_text()))


def fit_combiner(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    alphas: Mapping[str, float] = DEFAULT_ALPHAS,
    C: float = 1.0,
    info: dict | None = None,
) -> Combiner:
    """Fit on (X, y); pick thresholds on the human rows of (X_cal, y_cal), which must not
    overlap the fitting data or the guarantee is void."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=float)
    medians = np.nanmedian(np.where(np.isnan(X).all(axis=0), 0.0, X), axis=0)
    Xf = np.where(np.isnan(X), medians, X)
    mean, scale = Xf.mean(axis=0), Xf.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    lr.fit((Xf - mean) / scale, y)

    combiner = Combiner(
        features=list(features),
        medians=medians.tolist(),
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]),
        info=dict(info or {}),
    )
    human_scores = combiner.predict_matrix(np.asarray(X_cal)[np.asarray(y_cal) == 0])
    combiner.thresholds = {name: conformal_threshold(human_scores, a) for name, a in alphas.items()}
    combiner.info["alphas"] = dict(alphas)
    combiner.info["n_calibration_human"] = int(len(human_scores))
    return combiner
