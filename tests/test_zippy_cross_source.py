"""Small deterministic checks for the exploratory ZipPy evaluation helpers."""

import numpy as np

from eval.zippy_cross_source import _bigram_repetition, _bootstrap_intervals, _split_for_group, _threshold_at_fpr


def test_source_group_split_is_stable():
    assert _split_for_group("raid:source-1", 4) == _split_for_group("raid:source-1", 4)
    assert _split_for_group("raid:source-1", 4) != _split_for_group("raid:source-1", 5)


def test_one_percent_threshold_uses_only_the_allowed_human_tail():
    human_scores = np.arange(200, dtype=float)
    threshold = _threshold_at_fpr(human_scores)
    assert (human_scores > threshold).mean() == 0.01


def test_repeated_bigram_rate_is_deterministic():
    assert _bigram_repetition("a b a b a b") == 0.6
    assert _bigram_repetition("one two three") == 0.0


def test_group_bootstrap_intervals_are_reproducible():
    import pandas as pd

    frame = pd.DataFrame({"group_key": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    first = _bootstrap_intervals(frame, scores, 0.5, 100, 12)
    second = _bootstrap_intervals(frame, scores, 0.5, 100, 12)
    assert first == second
