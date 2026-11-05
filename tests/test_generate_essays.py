"""The essay generator against a fake OpenAI-compatible server on localhost."""

import contextlib
import io
import json
import os
import threading
import urllib.error
from email.message import Message
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
                self.reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

            def do_GET(self):
                self.reply(None)

            def reply(self, body):
                server.requests.append({"auth": self.headers.get("Authorization"), "body": body, "path": self.path})
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


def ollama_reply(text):
    return 200, {"message": {"role": "assistant", "content": text}, "done_reason": "stop"}


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "persuade.csv"
    pd.DataFrame({"essay_id_comp": ["e1", "e2"], "full_text": ["word " * 300, "word " * 200],
                  "prompt_name": ["Car-free cities", "Phones"], "grade_level": [8, None]}).to_csv(path, index=False)
    return path


def run(monkeypatch, argv):
    # Never read the real .env in tests: a real key must not reach the fake server or the internet.
    monkeypatch.setattr(generate_essays, "load_env", lambda: None)
    # Nor fall back to a real local model server; tests that want the fallback pass the flag again.
    monkeypatch.setattr("sys.argv", ["generate_essays", "--local-fallback", "", *argv])
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
                                         str(tmp_path / "o.jsonl"), "--base-url", server.url, "--n", "1",
                                         "--local-fallback", ""])
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


# The shape gemma3:4b replied in: a preamble behind a rule, a bold title, emphasis, then a question.
WRAPPED = """Okay, here\u2019s my essay \u2013 hopefully it\u2019s good!

---

**Should Drivers Be Allowed to Use Cell Phones? Absolutely Not.**

Seriously, letting drivers use their phones is *wrong*. A second is all it takes.

## Why it matters

People always say they're careful, but __nobody__ is careful enough.

---

Would you like me to make it sound younger, or add a counterargument?"""


def test_clean_essay_strips_the_chat_wrapper_and_markdown():
    assert generate_essays.clean_essay(WRAPPED) == (
        "Should Drivers Be Allowed to Use Cell Phones? Absolutely Not.\n\n"
        "Seriously, letting drivers use their phones is wrong. A second is all it takes.\n\n"
        "Why it matters\n\n"
        "People always say they're careful, but nobody is careful enough.")
    no_rules = ("Sure! Here is a persuasive letter to your principal:\n\nDear Principal,\n\n"
                "Service should be a choice.\n\nI hope this helps!\n\n(Word count: 9)")
    assert generate_essays.clean_essay(no_rules) == "Dear Principal,\n\nService should be a choice."


def test_clean_essay_keeps_openings_and_closings_that_belong_to_the_essay():
    essay = ("Here is why phones and driving don't mix.\n\nMy cousin crashed at 5 mph, 2 * 3 feet from home, "
             "while checking her user_name.\n\nI hope you agree.\n\nSincerely,\nA Student")
    assert generate_essays.clean_essay(essay) == essay
    assert generate_essays.clean_essay("Here is my essay:") == "Here is my essay:", "never empty"


def test_saved_essays_and_humanize_drafts_are_unwrapped(monkeypatch, tmp_path, csv):
    server = FakeServer([completion("Okay, here's my essay!\n\n---\n\n**Draft** body, longer than the chat around it."),
                         completion("Here is the rewritten essay:\n\nFinal *body*.")])
    out = tmp_path / "o.jsonl"
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--style", "humanize", "--n", "1",
                                 "--out", str(out), "--base-url", server.url]) == 0
    finally:
        server.close()
    rewrite_prompt = server.requests[1]["body"]["messages"][-1]["content"]
    assert rewrite_prompt.endswith("\n\nDraft body, longer than the chat around it.") and "Okay" not in rewrite_prompt
    assert json.loads(out.read_text())["text"] == "Final body."


def test_local_servers_get_a_longer_timeout(monkeypatch, tmp_path, csv):
    timeouts = []

    def fake_chat(*args, timeout, **kwargs):
        timeouts.append(timeout)
        return "Essay."

    monkeypatch.setattr(generate_essays, "chat", fake_chat)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    for i, url in enumerate(("http://localhost:11434/v1", "https://api.example.com/v1")):
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--n", "1",
                                 "--out", str(tmp_path / f"{i}.jsonl"), "--base-url", url]) == 0
    assert timeouts == [generate_essays.LOCAL_TIMEOUT, generate_essays.REMOTE_TIMEOUT]
    assert generate_essays.LOCAL_TIMEOUT > generate_essays.REMOTE_TIMEOUT


def test_spent_hosted_models_hand_the_rest_of_the_run_to_the_local_model(monkeypatch, tmp_path, csv, capsys):
    hosted = FakeServer([gemini_quota_error("19m15.12s"), gemini_quota_error("23h1m")])
    local = FakeServer([completion("Local one."), completion("Local two.")])
    out = tmp_path / "o.jsonl"
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--fallback-model", "lite", "--out", str(out),
                                 "--base-url", hosted.url, "--local-fallback", "gemma3:4b",
                                 "--local-url", local.url]) == 0
    finally:
        hosted.close()
        local.close()
    assert [r["body"]["model"] for r in hosted.requests] == ["m", "lite"], "the second essay goes straight to local"
    assert [r["body"]["model"] for r in local.requests] == ["gemma3:4b", "gemma3:4b"]
    assert all(r["auth"] is None for r in local.requests), "the hosted key never reaches the local server"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["text"] for r in rows} == {"Local one.", "Local two."}
    assert all(r["model"] == "gemma3:4b" and r["id"].endswith("-gemma3:4b-plain") for r in rows)
    assert "Switching to gemma3:4b" in capsys.readouterr().err


def test_unreachable_local_fallback_says_how_to_start_it(monkeypatch, tmp_path, csv, capsys):
    hosted = FakeServer([(404, {"error": {"message": "model 'm' does not exist"}})])
    try:
        code = run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--out", str(tmp_path / "o.jsonl"),
                                 "--base-url", hosted.url, "--n", "1", "--local-fallback", "gemma3:4b",
                                 "--local-url", "http://127.0.0.1:9/v1"])
    finally:
        hosted.close()
    err = capsys.readouterr().err
    assert code == 1 and "model 'm' does not exist" in err
    assert "could not reach http://127.0.0.1:9/v1" in err and "ollama serve" in err and "ollama pull gemma3:4b" in err


def test_app_fallback_tries_the_local_model_after_every_hosted_one():
    hosted = FakeServer([(401, {"error": "API key not valid"})])
    local = FakeServer([ollama_reply("From Gemma.")])
    try:
        text, model, errors = generate_essays.generate_with_fallback(
            hosted.url, "key", ["a", "b"], "Write.", 0.8, None, local.url, "gemma3:4b", 300, 16384)
    finally:
        hosted.close()
        local.close()
    assert (text, model) == ("From Gemma.", "gemma3:4b")
    assert len(hosted.requests) == 1, "a rejected key skips the other hosted models"
    assert len(errors) == 1 and "HTTP 401" in errors[0] and local.requests[0]["auth"] is None
    assert local.requests[0]["path"] == "/api/chat"
    assert local.requests[0]["body"]["options"] == {"temperature": 0.8, "num_predict": 300, "num_ctx": 16384}


def test_local_model_gets_a_window_for_the_whole_prompt_and_a_capped_answer():
    """Ollama's default window is 4,096 tokens, and it silently drops the start of a longer prompt."""
    local = FakeServer([ollama_reply("Short answer.")])
    system, prompt = "Style rule. " * 3000, "Shared document. " * 2000 + "The question?"
    try:
        assert generate_essays.chat_local(local.url, "gemma3:4b", prompt, 0.8, system,
                                          generate_essays.answer_token_limit(40)) == "Short answer."
    finally:
        local.close()
    options = local.requests[0]["body"]["options"]
    assert options["num_predict"] == 280
    assert options["num_ctx"] % 8192 == 0 and options["num_ctx"] >= (len(system) + len(prompt)) // 3 + 280
    assert [m["role"] for m in local.requests[0]["body"]["messages"]] == ["system", "user"]


def test_only_models_out_of_quota_for_hours_count_as_spent():
    hosted = FakeServer([gemini_quota_error("23h1m"), (503, {"error": "high demand"})])
    spent = set()
    try:
        with pytest.raises(generate_essays.APIError):
            generate_essays.generate_with_fallback(hosted.url, "key", ["a", "b"], "Write.", 0.8, None, spent=spent)
    finally:
        hosted.close()
    assert spent == {"a"}, "b was only busy"


def test_a_local_server_without_ollamas_chat_endpoint_gets_the_openai_request():
    local = FakeServer([(404, {"error": "Unexpected endpoint or method. (POST /api/chat)"}),
                        completion("From LM Studio.")])
    try:
        assert generate_essays.chat_local(local.url, "qwen", "Write.", 0.8) == "From LM Studio."
    finally:
        local.close()
    assert [r["path"] for r in local.requests] == ["/api/chat", "/v1/chat/completions"]


def test_a_local_model_ollama_lacks_is_an_error_not_a_second_request():
    local = FakeServer([(404, {"error": "model 'qwen3:4b' not found, try pulling it first"})])
    try:
        with pytest.raises(generate_essays.APIError, match="not found"):
            generate_essays.chat_local(local.url, "qwen3:4b", "Write.", 0.8)
    finally:
        local.close()
    assert len(local.requests) == 1


def test_app_fallback_without_a_working_local_model_reports_every_failure():
    hosted = FakeServer([(503, {"error": "busy"}), (503, {"error": "busy"})])
    try:
        with pytest.raises(generate_essays.APIError) as failed:
            generate_essays.generate_with_fallback(hosted.url, "key", ["a", "b"], "Write.", 0.8, None,
                                                   "http://127.0.0.1:9/v1", "gemma3:4b")
        with pytest.raises(generate_essays.APIError) as off:
            generate_essays.generate_with_fallback("http://127.0.0.1:9/v1", "key", ["a"], "Write.", 0.8, None)
    finally:
        hosted.close()
    assert "(a, b, gemma3:4b)" in str(failed.value) and "ollama serve" in str(failed.value)
    assert "(a)" in str(off.value), "no local model configured, none tried"


def test_unload_local_model_asks_ollama_to_drop_it_now():
    server = FakeServer([(200, {"done_reason": "unload"})])
    try:
        assert generate_essays.unload_local_model(server.url, "gemma3:4b") is True
    finally:
        server.close()
    assert server.requests[0]["path"] == "/api/generate"
    assert server.requests[0]["body"] == {"model": "gemma3:4b", "keep_alive": 0}
    assert generate_essays.unload_local_model("http://127.0.0.1:9/v1", "gemma3:4b") is False


def test_local_model_problem_says_what_to_run():
    tags = {"models": [{"name": "gemma3:4b"}, {"name": "llama3.2:latest"}]}
    server = FakeServer([(200, tags), (200, tags), (200, tags), (404, {"error": "not found"})])
    try:
        assert generate_essays.local_model_problem(server.url, "gemma3:4b") is None
        assert generate_essays.local_model_problem(server.url, "llama3.2") is None, "no tag means :latest"
        missing = generate_essays.local_model_problem(server.url, "qwen3:4b")
        assert "isn't downloaded" in missing and "ollama pull qwen3:4b" in missing
        assert generate_essays.local_model_problem(server.url, "any") is None, "a non-Ollama server is running"
    finally:
        server.close()
    assert server.requests[0]["path"] == "/api/tags"
    assert "ollama serve" in generate_essays.local_model_problem("http://127.0.0.1:9/v1", "gemma3:4b")


def test_missing_csv_and_missing_key_are_reported_up_front(monkeypatch, tmp_path, csv, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--persuade", str(tmp_path / "nope.csv"), "--model", "m", "--out", "o.jsonl"])
    assert "not found" in capsys.readouterr().err
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run(monkeypatch, ["--persuade", str(csv), "--out", "o.jsonl"])  # defaults: Gemini endpoint and key
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


PROFILE = {"style_profile": {
    "punctuation": {"em_dashes_frequency": "Never.",
                    "banned_ai_cliches": ["delve", "tapestry", "pivotal", "foster", "robust", "seamless",
                                          "myriad", "plethora", "beacon", "resonate"]},
    "system_prompt_overrides": {
        "negative_constraints": ["DO NOT use em-dashes anywhere."],
        "positive_constraints": ["MUST write 'wether' for 'whether'."],
        "golden_transformation_rule": "Write like an earnest student who types fast.",
    },
}}


def test_style_profile_is_the_system_prompt_of_every_call_and_joins_the_attack_label(monkeypatch, tmp_path, csv):
    profile = tmp_path / "mine.json"
    profile.write_text(json.dumps(PROFILE))
    server = FakeServer([completion("Draft."), completion("Humanized.")])
    out = tmp_path / "o.jsonl"
    try:
        assert run(monkeypatch, ["--persuade", str(csv), "--model", "m", "--style", "humanize", "--n", "1",
                                 "--style-profile", str(profile), "--out", str(out), "--base-url", server.url]) == 0
    finally:
        server.close()
    systems = [req["body"]["messages"][0] for req in server.requests]
    assert len(systems) == 2 and all(m["role"] == "system" for m in systems), "the rewrite keeps the style too"
    system = systems[0]["content"]
    golden, rules, reference = (system.index(s) for s in ("earnest student", "# Rules", "# Reference"))
    assert golden < rules < reference
    # Rules stay bullets however short they are; a long list of short phrases goes on one line.
    assert "- DO NOT use em-dashes anywhere." in system and "- MUST write 'wether'" in system
    assert "banned_ai_cliches: delve, tapestry, pivotal" in system and "golden_transformation_rule" not in system
    row = json.loads(out.read_text())
    assert row["text"] == "Humanized." and row["label"] == 1
    assert row["attack"] == "humanize+mine" and row["id"].endswith("-m-humanize+mine")


def test_profile_without_rules_is_all_reference():
    stats = {"sentences": {"mean": 18.3, "percentiles": [6, 16, 35]}, "vocabulary": {"mattr": 0.68}}
    system = generate_essays.style_instructions(stats)
    assert "# Rules" not in system and "## sentences" in system and "percentiles: 6, 16, 35" in system


def test_unreadable_style_profile_is_reported_up_front(monkeypatch, tmp_path, csv, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"casing": "quotes like "this" break JSON"}')
    for path in (bad, tmp_path / "missing.json"):
        with pytest.raises(SystemExit):
            run(monkeypatch, ["--persuade", str(csv), "--out", str(tmp_path / "o.jsonl"),
                              "--base-url", "http://127.0.0.1:9/v1", "--style-profile", str(path)])
        assert "could not read the style profile" in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()


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


def test_each_model_is_asked_for_a_thinking_level_it_has():
    level = generate_essays.thinking_level
    assert level("gemini-3-flash-preview", "minimal") == "minimal"
    assert level("gemini-3.8-flash", "minimal") == "low", "3.8 Flash starts at low"
    assert level("gemini-3-pro-preview", "medium") == "high"
    assert level("gemini-9-flash", "minimal") == "minimal", "an unknown model is asked as given"
    assert level("gemini-3.8-flash", None) is None


class FakeGemini:
    """Stands in for urllib's urlopen against Gemini: replies with the queued (status, body) pairs and
    records each request's body."""

    def __init__(self, monkeypatch, replies):
        self.replies, self.bodies = list(replies), []
        monkeypatch.setattr(generate_essays.urllib.request, "urlopen", self.urlopen)

    def urlopen(self, request, timeout):
        self.bodies.append(json.loads(request.data))
        status, body = self.replies.pop(0)
        payload = io.BytesIO(json.dumps(body).encode())
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "error", Message(), payload)
        return contextlib.closing(payload)


def gemini_output(text, usage=None):
    return 200, {"status": "completed", "usage": usage or {},
                 "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def test_gemini_gets_the_thinking_level_and_the_json_shape(monkeypatch):
    gemini = FakeGemini(monkeypatch, [gemini_output('{"blocks": []}', {"total_input_tokens": 900})])
    schema, usage = {"type": "object"}, {}
    assert generate_essays.chat(generate_essays.GEMINI_API_URL, "key", "gemini-3-flash-preview", "Write.", 0.8,
                                retries=1, thinking="low", json_schema=schema, usage=usage) == '{"blocks": []}'
    assert usage == {"total_input_tokens": 900}
    (body,) = gemini.bodies
    assert body["generation_config"] == {"temperature": 0.8, "thinking_level": "low"}
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": schema}


def test_a_model_that_turns_down_the_level_or_json_is_asked_again_without_it(monkeypatch):
    gemini = FakeGemini(monkeypatch, [
        (400, {"error": {"message": "thinking_level is not supported for this model."}}),
        (400, {"error": {"message": "Invalid value at 'response_format.schema'."}}),
        gemini_output("Written."),
    ])
    assert generate_essays.chat(generate_essays.GEMINI_API_URL, "key", "gemini-3.6-flash", "Write.", 0.8,
                                retries=1, thinking="low", json_schema={"type": "object"}) == "Written."
    first, second, third = gemini.bodies
    assert first["generation_config"] == {"thinking_level": "low"} and "response_format" in first
    assert "generation_config" not in second and "response_format" in second
    assert "generation_config" not in third and "response_format" not in third


def test_other_bad_requests_are_not_asked_again(monkeypatch):
    gemini = FakeGemini(monkeypatch, [(400, {"error": {"message": "API key not valid."}})])
    with pytest.raises(generate_essays.APIError, match="HTTP 400"):
        generate_essays.chat(generate_essays.GEMINI_API_URL, "key", "gemini-3.6-flash", "Write.", 0.8, retries=1,
                             thinking="low")
    assert len(gemini.bodies) == 1


def test_ollama_gets_the_json_shape_as_its_format():
    local = FakeServer([ollama_reply('{"blocks": []}')])
    try:
        generate_essays.chat_local(local.url, "gemma3:4b", "Write.", 0.8, json_schema={"type": "object"})
    finally:
        local.close()
    assert local.requests[0]["body"]["format"] == {"type": "object"}


REGISTERED = {
    "format": "registers-v2",
    "how_to_use": "Use one register at a time.",
    "registers": {
        "finalized": {"covers": "Essays and reports.", "habits": [
            {"habit": "Start with However.", "rate": "10%", "status": "established", "seen": "essays", "example": "x"}],
            "golden_rule": "Argue hard and plainly."},
        "notes": {"covers": "Lesson captures.", "habits": [
            {"habit": "Russian terms in Latin letters.", "rate": "rare", "status": "emerging", "seen": "2026-W44",
             "example": "tavar"}], "golden_rule": "Write fast for yourself."},
    },
    "shared": {"never": ["Em-dashes."], "banned_ai_cliches": ["delve"]},
}


def test_a_profile_with_registers_sends_only_the_one_asked_for():
    finished = generate_essays.style_instructions(REGISTERED, "finalized")
    assert "finished work register" in finished and "Argue hard and plainly." in finished
    assert "habit: Start with However.; rate: 10%; status: established" in finished, "a habit reads as one line"
    assert "Lesson captures" not in finished and "tavar" not in finished, "the notes register stays out"
    assert "# In every register" in finished and "Em-dashes." in finished and "delve" in finished
    notes = generate_essays.style_instructions(REGISTERED, "notes")
    assert "quick notes register" in notes and "tavar" in notes and "Essays and reports" not in notes
    assert generate_essays.style_instructions(REGISTERED) == finished, "the first register is the default"
    assert generate_essays.profile_registers(REGISTERED) == ["finalized", "notes"]
    assert generate_essays.profile_registers({"syntax_and_burstiness": {}}) == []


@pytest.mark.parametrize("profile", [REGISTERED, {"tone": "dry"}], ids=["registers", "older"])
def test_the_writers_passages_and_counts_come_last(profile):
    bare = generate_essays.style_instructions(profile, "notes")
    full = generate_essays.style_instructions(profile, "notes", ["first passage", "second passage"],
                                              ["About 5 sentences per 100 words."])
    assert full.startswith(bare), "the profile itself is unchanged"
    assert full[len(bare) + 1:].split("\n") == [
        "", "# The writer's own passages", generate_essays.EXAMPLES_INTRO,
        "", "Passage 1:", "first passage", "", "Passage 2:", "second passage",
        "", "# Counts to check", generate_essays.CHECKLIST_INTRO, "- About 5 sentences per 100 words."]


def test_a_slip_stays_at_the_writers_rate():
    sloppy = " ".join(f"The word{i} goes here , and it stops there ." for i in range(100))
    rates = {"space_before_punctuation": 2.0, "double_space": 2.0, "lowercase_i": 0.0}
    tempered = generate_essays.temper_quirks(sloppy, rates)
    # 900 words at 2 per 1,000 is 2 slips, out of the 200 the model wrote; the words don't change.
    assert 1 <= sum(tempered.count(mark) for mark in (" ,", " .")) <= 2
    assert tempered.replace(" ,", ",").replace(" .", ".") == sloppy.replace(" ,", ",").replace(" .", ".")
    assert generate_essays.temper_quirks(sloppy, rates) == tempered, "the same text keeps the same slips"


def test_tempering_only_removes_slips_and_leaves_normal_text_alone():
    text = "However i think its fine, i.e. it works (as planned).  Then it stopped ( for good )."
    assert generate_essays.temper_quirks(text, dict.fromkeys(("space_before_punctuation", "double_space",
                                                              "lowercase_i"), 0.0)) == \
        "However I think its fine, i.e. it works (as planned). Then it stopped (for good)."
    clean = "It works, and it stops. Then (as planned) it ends."
    assert generate_essays.temper_quirks(clean, dict.fromkeys(("space_before_punctuation", "double_space"), 500.0)) \
        == clean


def test_slip_rates_come_from_the_measurements_beside_the_profile(tmp_path):
    profile = tmp_path / "me_profile.json"
    assert generate_essays.quirk_rates(profile) == generate_essays.DEFAULT_QUIRK_RATES, "nothing measured"
    (tmp_path / "me").mkdir()
    (tmp_path / "me/measurements.json").write_text(json.dumps({
        "finalized_own": {"per_1000_words": {"space_before_punctuation": 2.0}},
        "stacked": {"per_1000_words": {"space_before_punctuation": 9.0}}}))
    assert generate_essays.quirk_rates(profile, "finalized")["space_before_punctuation"] == 2.0
    assert generate_essays.quirk_rates(profile, "notes")["space_before_punctuation"] == 9.0
    (tmp_path / "me/measurements.json").write_text(json.dumps({
        "finalized": {"whole": {"per_1000_words": {"space_before_punctuation": 0.4}}}}))  # the built-in profile's shape
    assert generate_essays.quirk_rates(profile, "finalized")["space_before_punctuation"] == 0.4


def test_a_habit_seen_once_and_another_registers_words_stay_out_of_the_prompt():
    profile = {"format": "registers-v2", "registers": {
        "finalized": {"habits": [
            {"habit": "Say clearly.", "rate": "2", "status": "established", "seen": "x", "example": "x"},
            {"habit": "Odd aside.", "rate": "1", "status": "once", "seen": "x", "example": "x"}]},
        "notes": {"habits": []}},
        "shared": {"signature_vocabulary": ["clearly (finalized: 5x)", "famously (notes: 16x)",
                                            "However (both registers)", "whatever (notes; once, 2026-W44)"]}}
    finished = generate_essays.style_instructions(profile, "finalized")
    assert "Say clearly." in finished and "Odd aside." not in finished
    assert "clearly (finalized" in finished and "However" in finished
    assert "famously" not in finished and "whatever" not in finished
    notes = generate_essays.style_instructions(profile, "notes")
    assert "famously" in notes and "clearly (finalized" not in notes and "whatever" not in notes


def test_a_recurring_misspelling_stays_at_the_share_the_writer_misspells():
    text = " ".join(["The soliders marched."] * 20)
    tempered = generate_essays.temper_misspellings(text, [("soliders", "soldiers", 0.1)])
    assert tempered.count("soliders") == 2 and tempered.count("soldiers") == 18
    assert generate_essays.temper_misspellings(text, [("soliders", "soldiers", 1.0)]) == text, "their only spelling"
    assert generate_essays.temper_misspellings("Cesar came.", [("cesar", "Caesar", 0.0)]) == "Caesar came."
    assert generate_essays.temper_misspellings("Teh end.", [("teh", "the", 0.0)]) == "The end.", "keeps the capital"
    clean = "The soldiers marched."
    assert generate_essays.temper_misspellings(clean, [("soliders", "soldiers", 1.0)]) == clean, "never adds one"


def test_misspelling_shares_are_counted_from_the_writing_beside_the_profile(tmp_path):
    profile = tmp_path / "me_profile.json"
    slips = {"recurring_misspellings": [
        "'soliders' for soldiers (3)", "'aswell' for as well", "'eg' and 'ie' with no full stops",
        "'Micronutreits' and 'Macronutreits' for micronutrients and macronutrients (reports)"]}
    profile.write_text(json.dumps({"format": "registers-v2", "registers": {"notes": {"slips": slips}}}))
    (tmp_path / "me").mkdir()
    (tmp_path / "me/corpus.jsonl").write_text(json.dumps({"text": "soliders and soldiers, soldiers, soldiers aswell."}))
    shares = {wrong: (right, share) for wrong, right, share in generate_essays.misspelling_shares(profile, "notes")}
    assert shares["soliders"] == ("soldiers", 0.25)
    assert shares["aswell"] == ("as well", 1.0), "written no other way"
    assert shares["Macronutreits"] == ("macronutrients", generate_essays.DEFAULT_SLIP_SHARE), "not in the writing"
    assert "eg" not in shares, "no word to put back"


def test_a_sentence_cut_off_to_fit_a_plan_is_joined_again():
    assert generate_essays._CUT_SHORT.sub(r"\1 ", "It fell. Another. piece of evidence.") == \
        "It fell. Another piece of evidence."
    assert generate_essays._CUT_SHORT.sub(r"\1 ", "Rome fell. Then it ended.") == "Rome fell. Then it ended."


GROQ = generate_essays.PROVIDERS["groq"].base_url
OPENROUTER = generate_essays.PROVIDERS["openrouter"].base_url


def test_groq_and_openrouter_keep_reasoning_out_and_ask_for_the_json_shape(monkeypatch):
    monkeypatch.delenv("OPENROUTER_DATA_COLLECTION", raising=False)
    server = FakeGemini(monkeypatch, [completion("{}")] * 3)
    schema = {"type": "object"}
    generate_essays.chat(GROQ, "key", "openai/gpt-oss-120b", "Write.", 0.8, thinking="minimal", json_schema=schema)
    generate_essays.chat(GROQ, "key", "llama-3.3-70b-versatile", "Write.", 0.8, thinking="low")
    generate_essays.chat(OPENROUTER, "key", "a/b:free", "Write.", 0.8, thinking="low", json_schema=schema)
    oss, llama, routed = server.bodies
    assert oss["include_reasoning"] is False and oss["reasoning_effort"] == "low", "Groq has no minimal"
    assert oss["response_format"]["json_schema"]["schema"] == schema
    assert "reasoning_effort" not in llama and "reasoning_format" not in llama, "llama doesn't reason"
    assert routed["reasoning"] == {"effort": "low", "exclude": True}
    assert routed["provider"] == {"data_collection": "deny"}, "a writer's passages skip hosts that keep prompts"


def test_openrouter_can_be_allowed_hosts_that_keep_prompts(monkeypatch):
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "allow")
    server = FakeGemini(monkeypatch, [completion("Written.")])
    generate_essays.chat(OPENROUTER, "key", "a/b:free", "Write.", 0.8)
    assert server.bodies[0]["provider"] == {"data_collection": "allow"}


def test_other_endpoints_get_a_plain_request(monkeypatch):
    server = FakeGemini(monkeypatch, [completion("Written.")])
    generate_essays.chat("https://api.example.com/v1", "key", "m", "Write.", 0.8, thinking="low",
                         json_schema={"type": "object"})
    assert set(server.bodies[0]) == {"model", "messages", "temperature"}


@pytest.mark.parametrize("message", ["This model does not support response format `json_schema`",
                                     "Failed to generate JSON", "Generated JSON does not match the expected schema."])
def test_a_model_turning_down_the_json_shape_is_asked_without_it(monkeypatch, message):
    server = FakeGemini(monkeypatch, [(400, {"error": {"message": message}}), completion("Written.")])
    reply = generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=1, json_schema={"type": "object"})
    assert reply == "Written."
    first, second = server.bodies
    assert "response_format" in first and "response_format" not in second


def test_a_model_turning_down_reasoning_is_asked_without_it(monkeypatch):
    server = FakeGemini(monkeypatch, [(400, {"error": {"message": "reasoning_format is not supported"}}),
                                      completion("Written.")])
    assert generate_essays.chat(GROQ, "key", "qwen/qwen3.8-27b", "Write.", 0.8, retries=1, thinking="low") == "Written."
    assert not {"reasoning_format", "reasoning_effort"} & set(server.bodies[1])


def test_any_other_bad_request_is_not_asked_again(monkeypatch):
    server = FakeGemini(monkeypatch, [(400, {"error": {"message": "reasoning is broken"}})])
    with pytest.raises(generate_essays.APIError, match="HTTP 400"):
        generate_essays.chat(GROQ, "key", "llama-3.3-70b-versatile", "Write.", 0.8, retries=1)
    assert len(server.bodies) == 1


@pytest.mark.parametrize(("reply", "text"), [("<think>Hmm.</think>\n\nThe essay.", "The essay."),
                                             ("<THINK>a\nb</THINK>The essay.", "The essay.")])
def test_reasoning_in_the_reply_is_never_saved_as_text(monkeypatch, reply, text):
    FakeGemini(monkeypatch, [completion(reply)])
    assert generate_essays.chat(GROQ, "key", "m", "Write.", 0.8) == text


def test_a_reply_that_is_only_reasoning_is_an_error(monkeypatch):
    FakeGemini(monkeypatch, [completion("<think>Still thinking")])
    with pytest.raises(generate_essays.APIError, match="empty response"):
        generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=1)


def test_a_reply_cut_off_at_its_token_limit_is_an_error(monkeypatch):
    FakeGemini(monkeypatch, [(200, {"choices": [{"message": {"content": "The essay stops"},
                                                 "finish_reason": "length"}]})])
    with pytest.raises(generate_essays.APIError, match="cut off"):
        generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=1)


def test_openrouter_reports_a_failed_host_inside_a_200(monkeypatch):
    FakeGemini(monkeypatch, [(200, {"error": {"code": 502, "message": "Provider disconnected"}})])
    with pytest.raises(generate_essays.APIError, match="Provider disconnected"):
        generate_essays.chat(OPENROUTER, "key", "m", "Write.", 0.8, retries=1)


def test_token_counts_come_back_under_geminis_field_names(monkeypatch):
    FakeGemini(monkeypatch, [(200, {"choices": [{"message": {"content": "Written."}}], "usage": {
        "prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100,
        "completion_tokens_details": {"reasoning_tokens": 4}}})])
    usage = {}
    generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, usage=usage)
    assert usage == {"total_tokens": 100, "total_input_tokens": 90, "total_output_tokens": 10,
                     "total_cached_tokens": 0, "total_thought_tokens": 4}


@pytest.mark.parametrize(("status", "message"), [
    (429, "Rate limit reached on tokens per day (TPD). Please try again in 1h2m3.4s."),
    (429, "Rate limit exceeded: free-models-per-day."),
    (402, "Insufficient credits"),
])
def test_a_spent_allowance_counts_as_out_of_quota(monkeypatch, status, message):
    FakeGemini(monkeypatch, [(status, {"error": {"message": message}})])
    with pytest.raises(generate_essays.QuotaExhausted):
        generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=1)


def test_a_short_wait_is_waited_out_at_the_time_groq_gives(monkeypatch):
    waits = []
    monkeypatch.setattr(generate_essays.time, "sleep", waits.append)
    FakeGemini(monkeypatch, [(429, {"error": {"message": "Please try again in 420ms."}}), completion("Written."),
                             (429, {"error": {"message": "Please try again in 1m2.5s."}}), completion("Written.")])
    generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=2)
    generate_essays.chat(GROQ, "key", "m", "Write.", 0.8, retries=2)
    assert waits == [pytest.approx(0.42), pytest.approx(62.5)]


def test_openrouter_errors_say_what_the_models_host_said(monkeypatch):
    FakeGemini(monkeypatch, [(429, {"error": {"message": "Provider returned error",
                                              "metadata": {"raw": "busy upstream"}}})])
    with pytest.raises(generate_essays.APIError, match="busy upstream"):
        generate_essays.chat(OPENROUTER, "key", "m", "Write.", 0.8, retries=1)


@pytest.mark.parametrize("status", [400, 403, 404])
def test_one_routed_models_refusal_moves_on_to_the_next(monkeypatch, status):
    FakeGemini(monkeypatch, [(status, {"error": {"message": "flagged"}}), completion("Written.")])
    text, model, errors = generate_essays.generate_with_fallback(OPENROUTER, "key", ["a", "b"], "Write.", 0.8, None)
    assert (text, model, len(errors)) == ("Written.", "b", 1)


def test_a_rejected_key_stops_at_the_first_routed_model(monkeypatch):
    server = FakeGemini(monkeypatch, [(401, {"error": {"message": "Invalid API Key"}})])
    with pytest.raises(generate_essays.APIError):
        generate_essays.generate_with_fallback(GROQ, "key", ["a", "b"], "Write.", 0.8, None)
    assert len(server.bodies) == 1


def test_providers_are_told_apart_by_address():
    assert generate_essays.provider_for("https://api.groq.com/openai/v1/") == "groq"
    assert generate_essays.provider_for(OPENROUTER) == "openrouter"
    assert generate_essays.provider_for(generate_essays.GEMINI_API_URL) == "gemini"
    assert generate_essays.provider_for("http://localhost:11434/v1") == "custom"


def test_listed_models_are_text_models_free_ones_first(monkeypatch):
    entries = [{"id": "z/paid", "architecture": {"output_modalities": ["text"]}},
               {"id": "a/paid-image", "architecture": {"output_modalities": ["image"]}},
               {"id": "m/free:free"}, {"id": "whisper-large-v3"}, {"id": "retired", "active": False}]
    monkeypatch.setattr(generate_essays.urllib.request, "urlopen", lambda request, timeout: contextlib.closing(
        io.BytesIO(json.dumps({"data": entries}).encode())))
    assert generate_essays.list_models(generate_essays.PROVIDERS["openrouter"], "key") == ["m/free:free", "z/paid"]


@pytest.mark.parametrize(("status", "accepted"), [(200, True), (401, False), (403, False), (500, None)])
def test_a_key_is_checked_on_the_path_only_a_good_key_opens(monkeypatch, status, accepted):
    urls = []

    def urlopen(request, timeout):
        urls.append(request.full_url)
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "error", Message(), io.BytesIO(b"{}"))
        return contextlib.closing(io.BytesIO(b"{}"))

    monkeypatch.setattr(generate_essays.urllib.request, "urlopen", urlopen)
    assert generate_essays.key_accepted(generate_essays.PROVIDERS["openrouter"], "key") is accepted
    assert urls == [f"{OPENROUTER}/key"], "OpenRouter's model list is public, so it can't check a key"


def test_every_provider_request_names_the_app_since_cloudflare_blocks_urllibs_default(monkeypatch):
    agents = []

    def urlopen(request, timeout):
        agents.append(request.get_header("User-agent"))
        return contextlib.closing(io.BytesIO(json.dumps(
            {"data": [], "choices": [{"message": {"content": "Written."}}]}).encode()))

    monkeypatch.setattr(generate_essays.urllib.request, "urlopen", urlopen)
    groq = generate_essays.PROVIDERS["groq"]
    generate_essays.chat(GROQ, "key", "m", "Write.", 0.8)
    generate_essays.list_models(groq, "key")
    generate_essays.key_accepted(groq, "key")
    assert agents == [generate_essays.USER_AGENT] * 3


class DroppedReply:
    """A response whose connection drops partway through."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        raise generate_essays.http.client.IncompleteRead(b"{")


def test_a_reply_lost_on_the_way_is_asked_for_again(monkeypatch):
    replies = [DroppedReply(), contextlib.closing(io.BytesIO(json.dumps(completion("Written.")[1]).encode()))]
    monkeypatch.setattr(generate_essays.urllib.request, "urlopen", lambda request, timeout: replies.pop(0))
    monkeypatch.setattr(generate_essays.time, "sleep", lambda seconds: None)
    assert generate_essays.chat(OPENROUTER, "key", "m", "Write.", 0.8, retries=2) == "Written."


def test_a_reply_lost_on_the_way_moves_on_to_the_next_model(monkeypatch):
    replies = [DroppedReply(), contextlib.closing(io.BytesIO(json.dumps(completion("Written.")[1]).encode()))]
    monkeypatch.setattr(generate_essays.urllib.request, "urlopen", lambda request, timeout: replies.pop(0))
    text, model, errors = generate_essays.generate_with_fallback(OPENROUTER, "key", ["a", "b"], "Write.", 0.8, None)
    assert (text, model, len(errors)) == ("Written.", "b", 1)


# -- several keys -----------------------------------------------------------------------


def keys_used(server):
    return [(request["body"]["model"], request["auth"]) for request in server.requests]


def test_a_key_pool_goes_round_its_keys_per_model():
    pool = generate_essays.KeyPool(["k1", "k2", "k1", ""])
    assert pool.keys == ["k1", "k2", ""] and len(pool) == 3
    assert [pool.next_key("a") for _ in range(4)] == ["k1", "k2", "", "k1"]
    pool.spend("k1", "a")
    assert [pool.next_key("a") for _ in range(2)] == ["k2", ""] and pool.next_key("b") == "k1", "spent on a only"
    pool.reject("k2")
    assert pool.next_key("a") == "" and pool.open_keys("a") == 1 and pool.open_keys("b") == 2
    notes = pool.take_notes()
    assert len(notes) == 2 and not any("k1" in note or "k2" in note for note in notes), "keys never in messages"
    assert pool.take_notes() == []


def test_a_model_runs_on_every_key_before_the_next_model():
    hosted = FakeServer([gemini_quota_error("23h1m"), completion("From key two.")])
    pool, spent = generate_essays.KeyPool(["k1", "k2"]), set()
    try:
        text, model, errors = generate_essays.generate_with_fallback(hosted.url, pool, ["a", "b"], "Write.", 0.8,
                                                                     None, spent=spent)
    finally:
        hosted.close()
    assert (text, model, errors, spent) == ("From key two.", "a", [], set())
    assert keys_used(hosted) == [("a", "Bearer k1"), ("a", "Bearer k2")]


def test_the_next_model_comes_only_when_every_key_is_spent_and_the_pool_remembers():
    hosted = FakeServer([gemini_quota_error("23h1m"), gemini_quota_error("23h1m"), completion("From b."),
                         completion("From b again.")])
    pool, spent = generate_essays.KeyPool(["k1", "k2"]), set()
    try:
        _, model, errors = generate_essays.generate_with_fallback(hosted.url, pool, ["a", "b"], "Write.", 0.8, None,
                                                                  spent=spent)
        _, again, _ = generate_essays.generate_with_fallback(hosted.url, pool, ["a", "b"], "Write.", 0.8, None)
    finally:
        hosted.close()
    assert (model, again, spent) == ("b", "b", {"a"}) and len(errors) == 1
    assert keys_used(hosted) == [("a", "Bearer k1"), ("a", "Bearer k2"), ("b", "Bearer k1"), ("b", "Bearer k2")]


def test_with_several_keys_a_per_minute_limit_waits_rather_than_moving_down(monkeypatch):
    monkeypatch.setattr(generate_essays, "RATE_MARGIN", 0)
    hosted = FakeServer([gemini_quota_error("0.2s"), gemini_quota_error("0.2s"), completion("Held.")])
    waits = []

    def pause(seconds, model):
        waits.append(model)
        generate_essays.time.sleep(seconds)

    try:
        text, model, _ = generate_essays.generate_with_fallback(
            hosted.url, generate_essays.KeyPool(["k1", "k2"]), ["a", "b"], "Write.", 0.8, None, pause=pause)
    finally:
        hosted.close()
    assert (text, model, waits) == ("Held.", "a", ["a"])
    assert [m for m, _ in keys_used(hosted)] == ["a", "a", "a"]


def test_one_key_still_moves_down_on_a_per_minute_limit():
    hosted = FakeServer([gemini_quota_error("20s"), completion("From b.")])
    try:
        _, model, _ = generate_essays.generate_with_fallback(
            hosted.url, "key", ["a", "b"], "Write.", 0.8, None, pause=lambda *_: pytest.fail("one key never waits"))
    finally:
        hosted.close()
    assert model == "b"


def test_repeated_per_minute_refusals_count_as_spent():
    pool = generate_essays.KeyPool(["k1", "k2"])
    for _ in range(generate_essays.RATE_STRIKES):
        pool.cool("k1", "a", 0)
    assert pool.open_keys("a") == 1 and pool.next_key("a") == "k2"


def test_a_rejected_key_is_left_out_and_the_others_carry_on():
    hosted = FakeServer([(400, {"error": "API key not valid. Please pass a valid API key."}), completion("One."),
                         completion("Two.")])
    pool = generate_essays.KeyPool(["bad", "good"])
    try:
        assert generate_essays.generate_with_fallback(hosted.url, pool, ["a"], "W.", 0.8, None)[0] == "One."
        assert generate_essays.generate_with_fallback(hosted.url, pool, ["a"], "W.", 0.8, None)[0] == "Two."
    finally:
        hosted.close()
    assert [auth for _, auth in keys_used(hosted)] == ["Bearer bad", "Bearer good", "Bearer good"]


def test_a_daily_limit_with_a_short_retry_is_still_a_spent_quota():
    hosted = FakeServer([(429, {"error": {"message": "Quota exceeded for quota id "
                                                     "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'. "
                                                     "Please retry in 31s."}}), completion("From b.")])
    spent = set()
    try:
        _, model, _ = generate_essays.generate_with_fallback(hosted.url, "key", ["a", "b"], "Write.", 0.8, None,
                                                             spent=spent)
    finally:
        hosted.close()
    assert (model, spent) == ("b", {"a"})
