"""The red-team loop with a stub Claude client and a stub detector: no network, no models."""

import json
from types import SimpleNamespace

import pytest

import align_engine
from opendetector import DetectorConfig, Report, SentenceScore

SENTENCES = ["The cat sat on the mat.", "It is important to delve into this topic.", "Then it slept."]
TEXT = " ".join(SENTENCES)


class StubClient:
    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.prompts, self.stop_reason = list(replies), [], stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        assert kwargs["fallbacks"] == "default" and align_engine.FALLBACK_BETA in kwargs["betas"]
        self.prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=SimpleNamespace(category="cyber"),
                               content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


class StubDetector:
    """Scores drafts from a fixed list; the middle sentence is always the most AI-like."""

    def __init__(self, scores):
        self.scores, self.config = list(scores), DetectorConfig()

    def analyze(self, text):
        sentences, pos = [], 0
        for i, s in enumerate(SENTENCES):
            start = TEXT.index(s, pos)
            sentences.append(SentenceScore(start, start + len(s), s, 0.9 if i == 1 else 0.1))
            pos = start + len(s)
        p = self.scores.pop(0)
        return Report(text=TEXT, words=150, verdict="likely_ai" if p > 0.9 else "likely_human",
                      ai_probability=p, calibrated=True, thresholds={"ai": 0.9, "review": 0.6},
                      sentences=sentences)


def test_returns_first_draft_when_already_under_target():
    client = StubClient(["draft one"])
    assert align_engine.generate_aligned_text("p", client=client, detector=StubDetector([0.01])) == "draft one"
    assert len(client.prompts) == 1


def test_rewrites_flagged_passages_until_under_target():
    client = StubClient(["draft one", "draft two", "draft three"])
    out = align_engine.generate_aligned_text("p", client=client, detector=StubDetector([0.95, 0.4, 0.02]))
    assert out == "draft three"
    retry = client.prompts[1]
    assert "<draft>\ndraft one\n</draft>" in retry
    assert "<passage>\nIt is important to delve into this topic.\n</passage>" in retry
    assert "The cat sat" not in retry.split("<draft>")[1].split("</draft>")[1], "only flagged passages are listed"


def test_returns_lowest_scoring_draft_when_target_never_reached():
    client = StubClient(["a", "b", "c"])
    out = align_engine.generate_aligned_text("p", max_attempts=3, client=client,
                                             detector=StubDetector([0.9, 0.3, 0.5]))
    assert out == "b"
    assert len(client.prompts) == 3, "no rewrite is requested after the last attempt"


def test_refusal_raises():
    with pytest.raises(align_engine.GenerationError, match="declined"):
        align_engine.generate_aligned_text("p", client=StubClient(["x"], stop_reason="refusal"),
                                           detector=StubDetector([0.5]))


def test_every_draft_is_logged_as_a_labelled_ai_sample(tmp_path):
    log_path = tmp_path / "attack.jsonl"
    align_engine.generate_aligned_text("p", client=StubClient(["a", "b"]), detector=StubDetector([0.8, 0.01]),
                                       log_path=log_path)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["text"] for r in rows] == ["a", "b"]
    assert all(r["label"] == 1 and r["dataset"] == "adaptive_attack" for r in rows)
    assert rows[0]["group_key"] == rows[1]["group_key"], "attempts on one prompt share a split"
    assert [r["attack"] for r in rows] == ["rewrite_0", "rewrite_1"]


def test_flagged_segments_merge_consecutive_sentences():
    report = StubDetector([0.5]).analyze(TEXT)
    report.sentences[2].ai_probability = 0.95
    assert align_engine.flagged_segments(report) == ["It is important to delve into this topic. Then it slept."]
