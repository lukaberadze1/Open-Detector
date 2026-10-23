"""The essay generator against a fake OpenAI-compatible server on localhost."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from eval import generate_essays


class FakeServer:
    """Replies with the queued (status, body) pairs in order and records each request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"auth": self.headers.get("Authorization"), "body": body})
                status, reply = server.replies.pop(0)
                payload = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def completion(text):
    return 200, {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "persuade.csv"
    pd.DataFrame({"essay_id_comp": ["e1", "e2"], "full_text": ["word " * 300, "word " * 200],
                  "prompt_name": ["Car-free cities", "Phones"], "grade_level": [8, None]}).to_csv(path, index=False)
    return path


def run(monkeypatch, argv):
    # Never read the real .env in tests: a real key must not reach the fake server or the internet.
    monkeypatch.setattr(generate_essays, "load_env", lambda: None)
    monkeypatch.setattr("sys.argv", ["generate_essays", *argv])
    monkeypatch.setattr(generate_essays.time, "sleep", lambda s: None)
    return generate_essays.main()


def test_writes_labelled_mirrors_without_a_key_on_localhost(monkeypatch, tmp_path, csv):
    server = FakeServer([completion("Essay one."), completion("Essay two.")])
    out = tmp_path / "mirrors.jsonl"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--style", "student",
                                 "--out", str(out), "--base-url", server.url]) == 0
    finally:
        server.close()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["text"] for r in rows} == {"Essay one.", "Essay two."}
    assert all(r["label"] == 1 and r["group_key"] in ("e1", "e2") for r in rows)
    assert all(req["auth"] is None for req in server.requests), "no empty Bearer header"
    prompts = [req["body"]["messages"][0]["content"] for req in server.requests]
    # pandas reads the grade column as floats (8.0, NaN); both must become sensible prompts.
    assert any("grade 8 student" in p for p in prompts) and any("grade 10 student" in p for p in prompts)
    assert not any("nan" in p or "8.0" in p for p in prompts)


def test_non_retryable_error_fails_fast_with_the_server_message(monkeypatch, tmp_path, csv, capsys):
    server = FakeServer([(404, {"error": {"message": "model 'm' does not exist"}})])
    try:
        code = run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--out", str(tmp_path / "o.jsonl"),
                                 "--base-url", server.url, "--n", "1"])
    finally:
        server.close()
    err = capsys.readouterr().err
    assert code == 1 and len(server.requests) == 1, "a 404 must not be retried"
    assert "HTTP 404" in err and "model 'm' does not exist" in err


def gemini_quota_error(retry_in):
    """The shape Gemini's OpenAI-compatible endpoint returns: a JSON list around the error."""
    return 429, [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                 "You exceeded your current quota.\n* Quota exceeded for metric: free_tier_requests, "
                 f"limit: 20, model: m\nPlease retry in {retry_in}."}}]


def test_spent_quota_switches_to_the_fallback_model(monkeypatch, tmp_path, csv, capsys):
    server = FakeServer([gemini_quota_error("19m15.12s"), completion("Fallback essay.")])
    out = tmp_path / "o.jsonl"
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--fallback-model", "lite",
                                 "--out", str(out), "--base-url", server.url, "--n", "1"]) == 0
    finally:
        server.close()
    assert [r["body"]["model"] for r in server.requests] == ["m", "lite"], "no pointless retries on a spent quota"
    row = json.loads(out.read_text())
    assert row["model"] == "lite" and row["id"].endswith("-lite-plain")
    assert "You exceeded your current quota" in capsys.readouterr().err


def test_spent_quota_without_fallback_stops_with_googles_message(monkeypatch, tmp_path, csv, capsys):
    server = FakeServer([gemini_quota_error("19m15.12s")])
    try:
        code = run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--fallback-model", "",
                                 "--out", str(tmp_path / "o.jsonl"), "--base-url", server.url, "--n", "1"])
    finally:
        server.close()
    err = capsys.readouterr().err
    assert code == 1 and len(server.requests) == 1
    assert "m is out of quota" in err and "Please retry in 19m15.12s" in err


def test_short_rate_limit_waits_as_asked_then_retries(monkeypatch, tmp_path, csv):
    waits = []
    server = FakeServer([gemini_quota_error("1.5s"), completion("Done.")])
    try:
        monkeypatch.setattr("sys.argv", ["g", "--persuade", str(csv), "--model", "m", "--out",
                                         str(tmp_path / "o.jsonl"), "--base-url", server.url, "--n", "1"])
        monkeypatch.setattr(generate_essays, "load_env", lambda: None)
        monkeypatch.setattr(generate_essays.time, "sleep", waits.append)
        assert generate_essays.main() == 0
    finally:
        server.close()
    assert waits == [1.5] and [r["body"]["model"] for r in server.requests] == ["m", "m"]


def test_server_errors_are_retried(monkeypatch, tmp_path, csv):
    server = FakeServer([(503, {"error": "busy"}), (429, {"error": "slow down"}), completion("Done.")])
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--out", str(tmp_path / "o.jsonl"),
                                 "--base-url", server.url, "--n", "1"]) == 0
    finally:
        server.close()
    assert len(server.requests) == 3


def test_empty_content_is_an_error_not_a_crash(monkeypatch, tmp_path, csv, capsys):
    server = FakeServer([(200, {"choices": [{"message": {"content": None}}]})])
    try:
        code = run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--out", str(tmp_path / "o.jsonl"),
                                 "--base-url", server.url, "--n", "1"])
    finally:
        server.close()
    assert code == 1 and "empty response" in capsys.readouterr().err


def test_missing_csv_and_missing_key_are_reported_up_front(monkeypatch, tmp_path, csv, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--persuade", str(tmp_path / "nope.csv"), "--model", "m", "--out", "o.jsonl"])
    assert "not found" in capsys.readouterr().err
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--persuade", str(csv), "--out", "o.jsonl"])  # defaults: Gemini endpoint and key
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_load_env_reads_dotenv_without_overriding(monkeypatch, tmp_path):
    from opendetector.env import load_env

    env = tmp_path / ".env"
    env.write_text("# comment\nexport GEMINI_API_KEY='from-file'\nOTHER_KEY=x=y\n\nbad line\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_KEY", "already-set")
    load_env(env)
    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["OTHER_KEY"] == "already-set", "exported variables win over the file"
    load_env(tmp_path / "missing.env")  # a missing file is fine
