"""The red-team report: saved samples scored with a stand-in detector."""

import json
from types import SimpleNamespace

import opendetector
from eval import redteam_report

PROSE = ("The hive sat behind the shed, and every summer the bees found their way into the garden. " * 12).strip()


def sample(sample_id, label, text=PROSE, **fields):
    return {"id": sample_id, "text": text, "label": label, "dataset": "local_redteam", **fields}


def test_the_report_counts_misses_and_false_flags_and_scores_each_sample_once(monkeypatch, tmp_path):
    verdicts = {"ai-caught": "likely_ai", "ai-missed": "likely_human", "human": "likely_ai"}
    scored = []

    class Detector:
        def analyze(self, text, sentences):
            scored.append(text)
            verdict = verdicts[text.split("|")[0]]
            return SimpleNamespace(verdict=verdict, ai_probability=0.9 if verdict == "likely_ai" else 0.1)

    monkeypatch.setattr(opendetector.Detector, "from_config", classmethod(lambda cls, config: Detector()))
    samples = tmp_path / "samples.jsonl"
    rows = [sample("1", 1, f"ai-caught|{PROSE}", thinking="low", temperature=0.8),
            sample("2", 1, f"ai-missed|{PROSE}", thinking="low", temperature=1.3),
            sample("3", 1, f"ai-missed|{PROSE}", thinking="minimal", temperature=1.3),
            sample("4", 0, f"human|{PROSE}"),
            sample("5", 1, "Too short to score."),
            sample("6", 1, f"human|{PROSE}", words_kept=98.0),  # a "rewrite" that handed back the original
            sample("1", 1, f"ai-caught|{PROSE}")]  # the same sample saved twice
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    argv = [str(samples), "--cache", str(tmp_path / "scores.jsonl"), "--report", str(tmp_path / "report.md")]

    assert redteam_report.main(argv) == 0
    text = (tmp_path / "report.md").read_text()
    assert "3 AI samples and 1 human ones" in text
    assert "**Missed (Likely human): 67%" in text and "Wrongly flagged (Likely AI): 100%" in text
    assert "### By temperature" in text and "| 1.3 | 2 | 100%" in text
    assert len(scored) == 4, "the short sample and the copy are left out, and the repeated one scored once"

    from eval.datasets import load_local

    assert "6" not in set(load_local(str(samples))["id"]), "nor does a copy reach the detector's training data"

    assert redteam_report.main(argv) == 0
    assert len(scored) == 4, "a second run reads the scores it cached"


def test_no_samples_long_enough_is_said(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(sample("1", 1, "Short.")) + "\n")
    assert redteam_report.main([str(samples), "--cache", str(tmp_path / "c.jsonl"),
                                "--report", str(tmp_path / "r.md")]) == 1
