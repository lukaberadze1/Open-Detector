import json
from types import SimpleNamespace

from eval import generate_essays
from eval.redteam_batch import meaning_problem, revise_blocks
from eval.trace import Trace, text_id


def _report(score):
    return SimpleNamespace(ai_probability=score, verdict="likely_ai", words=200, features={"desklib_mean": 1.0},
                           engines={}, windows=[{"start": 0, "end": 9, "desklib": 1.0}],
                           sentences=[SimpleNamespace(start=0, end=len("x"), ai_probability=score)])


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_versions_link_to_what_they_were_made_from(tmp_path):
    trace = Trace(tmp_path / "t.jsonl", setup="s")
    trace.bind(run="r1")
    source = trace.version("source", "one")
    trace.version("draft", "rejected", keep=False)
    made = trace.version("draft", "two", _report(0.9))
    assert source == text_id("one") and trace.head == made
    rows = _rows(tmp_path / "t.jsonl")
    assert [row["parent"] for row in rows] == [None, source, source]
    assert rows[2]["score"] == 0.9 and rows[2]["windows"][0]["desklib"] == 1.0 and rows[2]["run"] == "r1"


def test_a_prompt_is_stored_once_and_requests_point_to_it(tmp_path):
    trace = Trace(tmp_path / "t.jsonl")
    trace.request(prompt="same", system="profile", model="m")
    trace.request(prompt="same", system="profile", model="m")
    rows = _rows(tmp_path / "t.jsonl")
    assert [row["kind"] for row in rows] == ["blob", "blob", "request", "request"]
    assert rows[2]["prompt"] == rows[3]["prompt"] == text_id("same")


def test_chat_tells_the_observer_what_became_of_the_request_and_passes_tokens_on(monkeypatch):
    seen = []

    def inner(*args):
        usage = args[10]
        assert usage["want_thoughts"], "summaries are asked for only when something records them"
        usage.update(total_tokens=5, thought_summary="thought", retry_reasons=["HTTP 503"])
        return "reply"

    monkeypatch.setattr(generate_essays, "_chat", inner)
    monkeypatch.setattr(generate_essays, "OBSERVER", lambda **fields: seen.append(fields))
    usage = {}
    assert generate_essays.chat("u", "k", "m", "prompt", 0.8, usage=usage) == "reply"
    assert usage == {"total_tokens": 5}
    assert seen[0]["thoughts"] == "thought" and seen[0]["retries"] == ["HTTP 503"] and seen[0]["reply"] == "reply"


def test_chat_reports_a_failed_request_and_still_raises(monkeypatch):
    seen = []

    def inner(*_):
        raise generate_essays.APIError("overloaded")

    monkeypatch.setattr(generate_essays, "_chat", inner)
    monkeypatch.setattr(generate_essays, "OBSERVER", lambda **fields: seen.append(fields))
    try:
        generate_essays.chat("u", "k", "m", "prompt", 0.8)
    except generate_essays.APIError:
        pass
    else:
        raise AssertionError("should raise")
    assert "overloaded" in seen[0]["error"] and seen[0]["reply"] is None


def test_without_an_observer_nothing_extra_is_asked_for(monkeypatch):
    def inner(*args):
        assert args[10] is None
        return "reply"

    monkeypatch.setattr(generate_essays, "_chat", inner)
    monkeypatch.setattr(generate_essays, "OBSERVER", None)
    assert generate_essays.chat("u", "k", "m", "prompt", 0.8) == "reply"


def test_meaning_problem_names_the_reason():
    passage = "The fleet did not sail from Salamis in 480 and it would never have won there at all."
    assert meaning_problem(passage, passage) is None
    assert meaning_problem(passage, "The fleet sailed from Salamis in 480 and it would have won there at all.") == (
        "negation")
    assert meaning_problem(passage, "The fleet did not sail in 480.") in ("facts", "length")
    assert meaning_problem(passage, passage + " I was not there.") == "person"


def test_revising_records_each_rewrite_tried_and_why(tmp_path):
    def analyze(text, sentences=True):
        return _report(0.5 - 0.3 * ("bus" in text))

    def ask(prompt, pending):
        return json.dumps({"blocks": [{"id": "B1", "text": "A plain passage about the bus I missed yesterday."}]})

    trace = Trace(tmp_path / "t.jsonl")
    tidy = " ".join(f"Tidy sentence {n} offers a balanced overview of the topic." for n in range(3))
    outcomes = {}
    import collections

    outcomes = collections.Counter()
    revise_blocks({"a": tidy}, analyze, ask, str.strip, [], rounds=1, stats=outcomes, trace=trace)
    rows = _rows(tmp_path / "t.jsonl")
    stages = [row.get("stage") for row in rows if row["kind"] == "version"]
    assert stages == ["revise_start", "revise_round"]
    assert any(row["kind"] == "candidate" and row["status"] == "scored" and row["score"] < 0.5 for row in rows)
    assert any(row["kind"] == "passage" and row["kept"] for row in rows)
