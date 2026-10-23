"""The Streamlit page, run headless with stand-in engines (no model downloads), against the shipped
combiner."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

HUMAN = "My grandmother kept bees behind the shed, and every summer we got stung at least once. "
AI = "Furthermore, it is important to delve into the multifaceted tapestry of this topic. "
APP = str(Path(__file__).resolve().parent.parent / "app.py")


class KeywordClassifier:
    def __init__(self, name):
        self.name = name

    def window_logits(self, texts):
        return np.array([8.0 if "delve" in t else -8.0 for t in texts])


class KeywordBinoculars:
    name = "bino"

    def span_scores(self, text, spans):
        return np.array([0.6 if "delve" in text[a:b] else 1.1 for a, b in spans])


@pytest.fixture
def app(monkeypatch):
    import opendetector.engines as engines

    monkeypatch.setattr(engines, "build_engine",
                        lambda name, config: KeywordBinoculars() if name == "bino" else KeywordClassifier(name))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def analyze(at, text):
    at.text_area[0].input(text)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    return at, {m.label: m.value for m in at.metric}


def test_human_text_hides_its_score_and_highlights(app):
    at, metrics = analyze(app, HUMAN * 30)
    assert metrics["Verdict"] == "Likely human"
    assert metrics["AI likelihood"] == "*%"
    assert [e.label for e in at.expander] == ["Show sentence-level signal anyway", "What the detector measured"]


def test_ai_text_shows_score_category_and_signals(app):
    at, metrics = analyze(app, AI * 30)
    assert metrics["Verdict"] == "Likely AI"
    assert metrics["AI likelihood"].endswith("%") and metrics["AI likelihood"] != "*%"
    shown = [c.value for c in at.caption] + [i.value for i in at.info]
    assert any("paraphraser" in text for text in shown), "the AI-paraphrased category is reported"
    assert any("“delve” ×30" in m.value for m in at.markdown)
    assert "Where the signal is" in [h.value for h in at.subheader]
