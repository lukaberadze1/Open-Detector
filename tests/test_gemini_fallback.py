"""The Gemini website fallback: when it may run, what it sends, where it sits in the chain. No network."""

import pytest

from eval import gemini_fallback, generate_essays


@pytest.fixture
def cookies(monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_FALLBACK", "on")
    monkeypatch.setenv("OPENDETECTOR_PUBLIC", "off")
    monkeypatch.delenv("OPENDETECTOR_PACKAGED", raising=False)


def test_available_only_on_a_private_run_that_opted_in(cookies, monkeypatch):
    assert gemini_fallback.web_available()
    monkeypatch.setenv("OPENDETECTOR_PUBLIC", "on")
    assert not gemini_fallback.web_available()
    monkeypatch.setenv("OPENDETECTOR_PUBLIC", "off")
    monkeypatch.setenv("OPENDETECTOR_PACKAGED", "on")
    assert not gemini_fallback.web_available()
    monkeypatch.delenv("OPENDETECTOR_PACKAGED")
    monkeypatch.delenv("GEMINI_WEB_FALLBACK")
    assert not gemini_fallback.web_available()


def test_prompt_carries_system_and_schema():
    message = gemini_fallback.web_prompt("Write.", "Be plain.", {"type": "object"})
    assert message.startswith("Be plain.\n\nWrite.") and '{"type": "object"}' in message


def test_errors_never_carry_the_exception_text(cookies, monkeypatch):
    async def fails(message, timeout):
        raise RuntimeError("cookie one was rejected")

    monkeypatch.setattr(gemini_fallback, "_ask", fails)
    with pytest.raises(gemini_fallback.GeminiWebError) as caught:
        gemini_fallback.chat_web("Write.")
    assert "one" not in str(caught.value).replace("RuntimeError", "")


def test_json_reply_loses_its_fence(cookies, monkeypatch):
    async def fenced(message, timeout):
        return '```json\n{"a": 1}\n```'

    monkeypatch.setattr(gemini_fallback, "_ask", fenced)
    assert gemini_fallback.chat_web("Write.", json_schema={"type": "object"}) == '{"a": 1}'


def test_the_websites_error_reply_is_a_failure_not_a_sample(cookies, monkeypatch):
    async def apology(message, timeout):
        return "I encountered an error doing what you asked. Could you try again?"

    monkeypatch.setattr(gemini_fallback, "_ask", apology)
    with pytest.raises(gemini_fallback.GeminiWebError):
        gemini_fallback.chat_web("Write.")


def _hosted_fail(*args, **options):
    raise generate_essays.QuotaExhausted("a is out of quota")


def test_chain_tries_the_website_before_the_local_model(monkeypatch):
    monkeypatch.setattr(generate_essays, "chat", _hosted_fail)
    monkeypatch.setattr(generate_essays, "web_available", lambda: True)
    monkeypatch.setattr(generate_essays, "chat_web", lambda *args: "From the website.")
    local = lambda *args, **options: pytest.fail("the local model should not run")  # noqa: E731
    monkeypatch.setattr(generate_essays, "chat_local", local)
    text, model, errors = generate_essays.generate_with_fallback("http://x/v1", "k", ["a"], "Write.", 0.8, None,
                                                                 "http://l", "gemma3:4b")
    assert (text, model) == ("From the website.", gemini_fallback.WEB_MODEL) and len(errors) == 1


def test_chain_goes_on_to_the_local_model_when_the_website_fails(monkeypatch):
    def web_fails(*args):
        raise gemini_fallback.GeminiWebError("the Gemini website failed")

    monkeypatch.setattr(generate_essays, "chat", _hosted_fail)
    monkeypatch.setattr(generate_essays, "web_available", lambda: True)
    monkeypatch.setattr(generate_essays, "chat_web", web_fails)
    monkeypatch.setattr(generate_essays, "chat_local", lambda *args, **options: "From Gemma.")
    text, model, errors = generate_essays.generate_with_fallback("http://x/v1", "k", ["a"], "Write.", 0.8, None,
                                                                 "http://l", "gemma3:4b")
    assert (text, model) == ("From Gemma.", "gemma3:4b") and len(errors) == 2
