"""Pipeline tests with stand-in engines, so no model downloads are needed."""

import numpy as np

from opendetector import Detector, DetectorConfig
from opendetector.combiner import fit_combiner

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
    assert set(feats) == {"fake_mean", "fake_max", "bino_doc", "bino_min"} == set(det.feature_names)
    assert feats["fake_max"] > feats["fake_mean"]
    assert feats["bino_min"] == 0.7


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


def test_highlights_do_not_bleed_across_a_paragraph_break():
    det = make()
    report = det.analyze(HUMAN * 6 + "\n\n" + AI * 6)
    human_part = [s.ai_probability for s in report.sentences[:6]]
    ai_part = [s.ai_probability for s in report.sentences[6:]]
    # Every human sentence scores like pure human text, even the one next to the AI paragraph.
    assert max(human_part) < min(ai_part)
    assert human_part[-1] == human_part[0]


def test_poetry_and_lists_are_not_scored():
    det = make(min_words=20)
    poem = "\n".join(["the frost is on the field", "the crows are gone", "I walk the long road home",
                      "past the empty barn", "where my father stood", "and the light goes down"] * 2)
    report = det.analyze(poem)
    assert report.verdict == "not_prose" and report.ai_probability is None
    assert det.classifiers[0].calls == [], "no model runs on text it won't score"


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
