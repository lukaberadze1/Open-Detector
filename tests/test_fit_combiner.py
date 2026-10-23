import json
import math

import numpy as np
import pandas as pd
import pytest

from eval.fit_combiner import MIN_PARAPHRASE_ROWS, fit_paraphrase_head, rewritten, sparse_features
from opendetector.combiner import Combiner


def test_rewrite_labels():
    assert rewritten("synonym") == rewritten("paraphrase") == rewritten("humanize") == rewritten("rewrite_2") == 1
    assert rewritten("none") == rewritten("student") == rewritten("rewrite_0") == 0
    assert math.isnan(rewritten("homoglyph")) and math.isnan(rewritten("zero_width_space"))


def test_features_a_dataset_lacks_are_left_out():
    df = pd.DataFrame({"dataset": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "new": [1, 2, np.nan, np.nan],
                       "flaky": [1, np.nan, 3, 4]})
    assert sparse_features(df, ["x", "new", "flaky"]) == ["new"]


def ai_rows(n_rewritten, n_as_written, rng):
    n = n_rewritten + n_as_written
    attack = ["synonym"] * n_rewritten + ["none"] * n_as_written
    signal = np.r_[rng.normal(2, 1, n_rewritten), rng.normal(0, 1, n_as_written)]
    return pd.DataFrame({"label": 1, "attack": attack, "group_key": [f"g{i}" for i in range(n)],
                         "spin": signal, "noise": rng.normal(size=n)})


def test_paraphrase_head_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    dev = pd.concat([ai_rows(150, 150, rng), pd.DataFrame({"label": [0] * 5, "attack": "synonym",
                     "group_key": "h", "spin": 9.0, "noise": 0.0})], ignore_index=True)
    head, n_rw, n_aw = fit_paraphrase_head(dev, ["spin", "noise"], seed=0)
    assert (n_rw, n_aw) == (150, 150), "human rows never train the head"
    assert head.coef[0] > abs(head.coef[1])
    test = ai_rows(500, 500, rng)
    p = head.predict_matrix(test[["spin", "noise"]].values)
    called = p > head.thresholds["paraphrased"]
    assert called[test.attack == "none"].mean() < 0.1  # the threshold targets 5%

    main = Combiner(["spin"], [0.0], [0.0], [1.0], [1.0], 0.0, {"ai": 0.9, "review": 0.5})
    main.paraphrase = vars(head)
    main.save(tmp_path / "c.json")
    loaded = Combiner.load(tmp_path / "c.json").paraphrase_head
    np.testing.assert_allclose(loaded.predict_matrix(test[["spin", "noise"]].values), p)


def test_paraphrase_head_needs_enough_of_each_kind():
    head, n_rw, _ = fit_paraphrase_head(ai_rows(MIN_PARAPHRASE_ROWS - 1, 100, np.random.default_rng(1)),
                                           ["spin"], seed=0)
    assert head is None and n_rw == MIN_PARAPHRASE_ROWS - 1


def test_combiners_saved_before_the_head_still_load(tmp_path):
    old = {"features": ["a"], "medians": [0.0], "mean": [0.0], "scale": [1.0], "coef": [1.0], "intercept": 0.0,
           "thresholds": {"ai": 0.9, "review": 0.5}, "info": {}}
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = Combiner.load(tmp_path / "old.json")
    assert loaded.paraphrase_head is None
    assert loaded.predict_proba([{"a": 0.0}])[0] == pytest.approx(0.5)
