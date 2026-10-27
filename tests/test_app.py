"""The Streamlit app run headless: stand-in engines (no model downloads) against the shipped
combiners, and a stand-in text model on the Red team page writing to a temporary folder."""

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HUMAN = "My grandmother kept bees behind the shed, and every summer we got stung at least once. "
AI = "Furthermore, it is important to delve into the multifaceted tapestry of this topic. "


class KeywordClassifier:
    def __init__(self, name):
        self.name = name

    def window_logits(self, texts):
        return np.array([8.0 if "delve" in t else -8.0 for t in texts])


class KeywordBinoculars:
    name = "bino"

    def span_scores(self, text, spans):
        return np.array([0.6 if "delve" in text[a:b] else 1.1 for a, b in spans])


@pytest.fixture(autouse=True)
def stand_in_engines(monkeypatch):
    import opendetector.engines as engines
    from ui import models

    monkeypatch.setattr(engines, "build_engine",
                        lambda name, config: KeywordBinoculars() if name == "bino" else KeywordClassifier(name))
    models.release()  # engines another test loaded
    yield
    models.release()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def analyze(text, profile="full"):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["profile"] = profile
    at.run()
    at.text_area(key="keep:detect:text").input(text)
    button(at, "Analyze").click()
    at.run()
    assert not at.exception, at.exception
    return at, at.session_state["detect:result"]


def test_ai_text_is_flagged_with_every_engine(monkeypatch):
    at, result = analyze(AI * 30)
    assert result["report"].verdict == "likely_ai"
    assert set(result["report"].engines) == {"glyph", "desklib", "bino"}
    assert "likely_ai" in result["bands"]
    assert [tab.label for tab in at.tabs] == ["Highlights", "Sentences", "Details"]


def test_human_text_on_the_slim_build_runs_desklib_alone():
    _, result = analyze(HUMAN * 30, profile="slim")
    assert result["report"].verdict == "likely_human"
    assert set(result["report"].engines) == {"desklib"}


def test_short_text_abstains():
    at, result = analyze(HUMAN * 3)
    assert result["report"].verdict == "insufficient_text"
    assert any("Need at least 150 words" in info.value for info in at.info)


def test_switched_off_engine_is_left_out():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["profile"] = "full"
    at.session_state["keep:detect:engine:bino"] = False
    at.run()
    at.text_area(key="keep:detect:text").input(AI * 30)
    button(at, "Analyze").click()
    at.run()
    result = at.session_state["detect:result"]
    assert set(result["report"].engines) == {"glyph", "desklib"}
    assert result["missing"] == ["bino_doc", "bino_min"]


def test_public_deploy_leaves_out_the_red_team_page(monkeypatch):
    from ui import nav

    built = []
    monkeypatch.setattr(nav, "redteam_page", lambda: built.append("red team"))
    monkeypatch.setenv("OPENDETECTOR_REDTEAM", "off")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert built == []


def red_team_page():
    from ui import redteam

    redteam.render()


def test_red_team_sample_is_recorded_as_ai_with_its_score(monkeypatch, tmp_path):
    from ui import redteam

    for name, path in (("ROOT", tmp_path), ("OUTPUT_PATH", tmp_path / "local_redteam.jsonl"),
                       ("EXPORT_DIR", tmp_path / "exports"), ("BATCH_DIR", tmp_path / "batches"),
                       ("STYLE_PATH", tmp_path / "no_style_profile.json")):
        monkeypatch.setattr(redteam, name, path)
    monkeypatch.setattr(redteam, "load_env", lambda: None)  # never read the real .env
    monkeypatch.setattr(redteam, "local_fallback_problem", lambda url, model: None)
    monkeypatch.setattr(redteam, "generate_with_fallback", lambda *args, **kwargs: (AI * 30, "stub-model", []))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    at = AppTest.from_function(red_team_page, default_timeout=60)
    at.session_state["profile"] = "slim"
    at.run()
    assert not at.exception, at.exception
    at.text_area(key="keep:redteam:prompt").input("An essay on keeping bees.")
    button(at, "Generate sample").click()
    at.run()
    assert not at.exception, at.exception

    rows = [json.loads(line) for line in (tmp_path / "local_redteam.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0] | {"text": None, "id": None, "prompt": None, "group_key": None, "ai_probability": None} == {
        "id": None, "text": None, "label": 1, "dataset": "local_redteam", "domain": "custom_prompt",
        "model": "stub-model", "attack": "prompt+plain", "group": "ai", "group_key": None, "prompt": None,
        "source_file": None, "target_section": None, "ai_probability": None, "verdict": "likely_ai",
        "calibrated": True, "detector_profile": "slim",
    }
    assert rows[0]["ai_probability"] > 0.9
    assert len(list((tmp_path / "exports").glob("AI_REDTEAM_prompt_*.txt"))) == 1
    assert at.session_state["last_generated_result"]["detect_text"] == rows[0]["text"]
