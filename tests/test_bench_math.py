"""Stopping and resuming the math benchmark, and scoring in batches. No models."""

import json
from types import SimpleNamespace

import numpy as np

from eval.bench_math import finished_rows, settings_of
from eval.math_attack import CachedScorer
from opendetector.segment import split_sentences


def test_resume_reads_only_rows_with_the_same_settings_and_survives_a_cut_off_line(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [{"source_text": "a", "attack": "math+contextual+t5", "text": "x"},
            {"source_text": "b", "attack": "math+contextual", "text": "y"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"source_text": "c', encoding="utf-8")
    assert list(finished_rows(path, "math+contextual+t5")) == ["a"]
    assert finished_rows(tmp_path / "missing.jsonl", "any") == {}


def test_quick_runs_are_kept_apart_from_full_ones():
    args = SimpleNamespace(contextual=True, power=False, paraphraser="t5", quick=True)
    assert settings_of(args) == "math+contextual+t5+quick"
    assert settings_of(SimpleNamespace(contextual=False, power=False, paraphraser=None, quick=False)) == "math+guided"


class Engine:
    name = "stub"

    def __init__(self):
        self.calls = []

    def window_logits(self, texts):
        self.calls.append(list(texts))
        return np.zeros(len(texts))


def test_many_runs_each_new_window_once_in_one_batch():
    engine = Engine()
    config = SimpleNamespace(doc_window_words=350, doc_stride_words=175)
    detector = SimpleNamespace(
        binoculars=None, combiner=object(), config=config, classifiers=[engine],
        _prepare=lambda text: (text, None, split_sentences(text)), _doc_features=lambda scores, windows, extra: 0,
        _probabilities=lambda rows: [0.5])
    scorer = CachedScorer(detector)
    assert scorer.many(["One whole sentence here.", "Another whole sentence here.", "One whole sentence here."]) \
        == [0.5, 0.5, 0.5]
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 2
