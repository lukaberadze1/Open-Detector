import pandas as pd

from eval.trial_compare import compare
from opendetector.combiner import Combiner


def _combiner(coef: float) -> Combiner:
    return Combiner(features=["desklib_mean"], medians=[0.0], mean=[0.0], scale=[1.0], coef=[coef], intercept=0.0,
                    thresholds={"ai": 0.8, "review": 0.15}, info={})


def test_compare_counts_each_combiner_on_the_same_texts():
    frame = pd.DataFrame({"desklib_mean": [3.0, 0.5, -3.0], "label": [1, 1, 0]})
    report = compare(_combiner(1.0), _combiner(0.1), {"set": frame})
    assert "| set | 2 | 1 | shipped | 50% | 0% | 0% | 100% | 1.000 |" in report
    assert "| set | 2 | 1 | trial | 0% | 0% | 0% | 0% | 1.000 |" in report


def test_compare_handles_a_set_with_one_label():
    report = compare(_combiner(1.0), _combiner(1.0), {"ai": pd.DataFrame({"desklib_mean": [3.0], "label": [1]})})
    assert "| ai | 1 | 0 | trial | 100% | 0% | - | - | - |" in report


def test_a_redteam_cache_splits_by_guidance():
    from eval.trial_compare import by_guidance

    frame = pd.DataFrame({"attack": ["full+revise3-sentences", "prompt+task", "x+math", "humanizer+cive-bart+best8"],
                          "label": [1, 1, 1, 1]})
    parts = by_guidance("saved", frame)
    assert len(parts["saved, guided"]) == 2 and len(parts["saved, not guided"]) == 1
    assert len(parts["saved, public humanizers"]) == 1
