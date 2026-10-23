import numpy as np
import pytest

from opendetector.combiner import Combiner, fit_combiner
from opendetector.metrics import auroc, conformal_threshold, tpr_at_fpr, wilson_interval


def test_auroc_matches_definition_with_ties():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_tpr_at_fpr_respects_the_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)]
    tpr, t = tpr_at_fpr(y, s, 0.01)
    assert (s[y == 0] > t).mean() <= 0.01
    assert 0.65 < tpr < 0.85  # P(N(3,1) > z_0.99 ≈ 2.33) ≈ 0.75


def test_conformal_threshold_controls_false_positives_on_new_data():
    rng = np.random.default_rng(1)
    t = conformal_threshold(rng.normal(size=2000), alpha=0.05)
    assert (rng.normal(size=200_000) > t).mean() <= 0.06


def test_conformal_threshold_refuses_when_calibration_set_is_too_small():
    assert conformal_threshold(np.array([0.1, 0.2, 0.3]), alpha=0.01) == 1.0


def test_wilson_interval_contains_rate():
    lo, hi = wilson_interval(5, 1000)
    assert lo < 0.005 < hi and lo > 0


def make_data(n, rng):
    y = rng.integers(0, 2, n)
    X = np.c_[y * 2 + rng.normal(size=n), -y + rng.normal(size=n), rng.normal(size=n)]
    return X, y


def test_combiner_learns_signal_and_round_trips(tmp_path):
    rng = np.random.default_rng(2)
    X, y = make_data(4000, rng)
    X_cal, y_cal = make_data(2000, rng)
    X[rng.random(X.shape) < 0.05] = np.nan  # missing engine outputs get imputed
    comb = fit_combiner(X, y, ["a", "b", "noise"], X_cal, y_cal)

    X_test, y_test = make_data(2000, rng)
    p = comb.predict_matrix(X_test)
    assert auroc(y_test, p) > 0.9
    assert comb.coef[0] > 0 and comb.coef[1] < 0 and abs(comb.coef[2]) < abs(comb.coef[1])
    assert comb.thresholds["ai"] > comb.thresholds["review"]
    assert (p[y_test == 0] > comb.thresholds["ai"]).mean() < 0.02

    comb.save(tmp_path / "c.json")
    loaded = Combiner.load(tmp_path / "c.json")
    np.testing.assert_allclose(loaded.predict_matrix(X_test), p)
    row = dict(zip(["a", "b", "noise"], X_test[0], strict=True))
    assert loaded.predict_proba([row])[0] == pytest.approx(p[0])
    assert np.isfinite(loaded.predict_proba([{"a": 1.0}])[0]), "absent features fall back to medians"
