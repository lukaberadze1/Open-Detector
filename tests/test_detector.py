"""Pipeline tests with stand-in engines, so no model downloads are needed."""

from dataclasses import asdict

import numpy as np
import pytest

from opendetector import Detector, DetectorConfig
from opendetector.combiner import fit_combiner, fit_logistic, set_thresholds

HUMAN = "My grandmother kept bees behind the shed, and every summer we got stung at least once. "
AI = "Furthermore, it is important to delve into the multifaceted tapestry of this topic. "


class KeywordClassifier:
    """Log-odds rise with the share of 'delve'/'tapestry' sentences."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def window_logits(self, texts):
        self.calls.append(list(texts))
        return np.array([4.0 * t.count("delve") / max(t.count("."), 1) - 2.0 for t in texts])


class LengthBinoculars:
    name = "bino"

    def span_scores(self, text, spans):
        return np.array([0.7 if "delve" in text[a:b] else 1.0 for a, b in spans])


def make(min_words=40, combiner=None):
    config = DetectorConfig(min_words=min_words, doc_window_words=120, doc_stride_words=60,
                            local_window_words=40, local_stride_words=20)
    return Detector([KeywordClassifier()], LengthBinoculars(), combiner, config)


def test_feature_names_and_values():
    det = make()
    feats = det.features(HUMAN * 6 + AI * 6)
    assert set(feats) == {"fake_mean", "fake_max", "bino_doc", "bino_min", "bino_burst",
                          "style_len_cv", "style_mattr", "style_aiisms"} == set(det.feature_names)
    assert feats["fake_max"] > feats["fake_mean"]
    assert feats["bino_min"] == 0.7
    # Six sentences score 1.0 and six 0.7, so the sentence-to-sentence spread is 0.15.
    assert feats["bino_burst"] == pytest.approx(0.15)


def test_stylometry_can_be_switched_off():
    det = make()
    det.config.stylometry = False
    assert not any(f.startswith("style_") for f in det.features(HUMAN * 6))
    assert not any(f.startswith("style_") for f in det.feature_names)


def test_burstiness_needs_enough_sentences():
    assert np.isnan(make().features(HUMAN * 3)["bino_burst"])


def test_short_text_abstains_without_running_models():
    det = make(min_words=200)
    report = det.analyze(HUMAN * 2)
    assert report.verdict == "insufficient_text" and report.ai_probability is None
    assert det.classifiers[0].calls == []


def test_long_text_is_fully_scored_not_truncated():
    det = make()
    text = HUMAN * 30 + AI * 5  # the AI part sits at the very end
    report = det.analyze(text)
    scored = "".join(det.classifiers[0].calls[0])
    assert "delve" in scored, "the tail of a long document must reach the classifier"
    assert report.sentences[-1].ai_probability > report.sentences[0].ai_probability


def test_sentence_scores_follow_the_ai_passage():
    det = make()
    report = det.analyze(HUMAN * 8 + AI * 8 + HUMAN * 8)
    probs = [s.ai_probability for s in report.sentences]
    assert len(report.sentences) == 24
    assert max(probs[8:16]) > max(probs[:4]) and max(probs[8:16]) > max(probs[-4:])
    assert all(report.text[s.start : s.end] == s.text for s in report.sentences)


def test_uncalibrated_and_calibrated_verdicts():
    det = make()
    assert not det.analyze(HUMAN * 10).calibrated

    rng = np.random.default_rng(0)
    texts = [(HUMAN * int(rng.integers(6, 12)), 0) for _ in range(150)] + \
            [(HUMAN * int(rng.integers(0, 3)) + AI * int(rng.integers(6, 12)), 1) for _ in range(150)]
    X = np.array([[det.features(t)[f] for f in det.feature_names] for t, _ in texts])
    y = np.array([label for _, label in texts])
    det.combiner = fit_combiner(X, y, det.feature_names, X, y)

    human, ai = det.analyze(HUMAN * 10), det.analyze(AI * 10)
    assert human.calibrated and human.verdict == "likely_human"
    assert ai.verdict == "likely_ai" and ai.ai_probability > human.ai_probability


def test_hidden_characters_are_reported():
    report = make().analyze((HUMAN * 6).replace("bees", "be​es"))
    assert report.normalization.invisible_chars == 6
    assert "be​es" not in report.text


def fitted(det):
    rng = np.random.default_rng(0)
    texts = [(HUMAN * int(rng.integers(6, 12)), 0) for _ in range(150)] + \
            [(HUMAN * int(rng.integers(0, 3)) + AI * int(rng.integers(6, 12)), 1) for _ in range(150)]
    X = np.array([[det.features(t)[f] for f in det.feature_names] for t, _ in texts])
    y = np.array([label for _, label in texts])
    return fit_combiner(X, y, det.feature_names, X, y), X, y


def test_paraphrase_head_is_only_asked_about_ai_text():
    det = make()
    det.combiner, X, y = fitted(det)
    assert det.analyze(AI * 10).paraphrase_probability is None, "no head fitted, no category"

    # A head that calls text 'paraphrased' when it has the fewest 'delve' sentences.
    head = fit_logistic(X[y == 1], (X[y == 1, 0] < np.median(X[y == 1, 0])).astype(int), det.feature_names)
    det.combiner.paraphrase = asdict(set_thresholds(head, head.predict_matrix(X[y == 1]), {"paraphrased": 0.5}))
    ai = det.analyze(AI * 10)
    threshold = det.combiner.paraphrase_head.thresholds["paraphrased"]
    assert ai.verdict == "likely_ai" and 0 <= ai.paraphrase_probability <= 1
    assert ai.ai_paraphrased == (ai.paraphrase_probability > threshold)
    human = det.analyze(HUMAN * 10)
    assert human.paraphrase_probability is None and human.ai_paraphrased is None
