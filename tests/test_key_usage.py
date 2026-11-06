"""Counting each Gemini key's requests per model per quota day (eval/key_usage.py)."""

from datetime import UTC, datetime

import pytest

from eval import generate_essays, generation_settings, key_usage

MORNING = datetime(2026, 11, 6, 17, 0, tzinfo=UTC)  # 09:00 in California, 6 November
NEXT_DAY = datetime(2026, 11, 7, 9, 0, tzinfo=UTC)  # 01:00 in California, 7 November
GEMINI = "https://generativelanguage.googleapis.com"


def test_requests_are_counted_per_key_and_model_and_the_file_holds_no_key(tmp_path):
    path = tmp_path / "usage.json"
    for _ in range(3):
        key_usage.record("secret-one", "flash", path=path, now=MORNING)
    key_usage.record("secret-two", "flash", path=path, now=MORNING)
    key_usage.record("secret-two", "lite", spent=True, path=path, now=MORNING)
    one = key_usage.left("secret-one", ["flash", "lite"], {"flash": 20}, path=path, now=MORNING)
    two = key_usage.left("secret-two", ["flash", "lite"], {"flash": 20}, path=path, now=MORNING)
    assert one == [{"model": "flash", "used": 3, "limit": 20, "left": 17, "spent": False},
                   {"model": "lite", "used": 0, "limit": None, "left": None, "spent": False}]
    assert two[0]["left"] == 19 and two[1] == {"model": "lite", "used": 0, "limit": None, "left": 0, "spent": True}
    assert "secret" not in path.read_text()


def test_counts_start_again_at_midnight_pacific(tmp_path):
    path = tmp_path / "usage.json"
    key_usage.record("k", "flash", spent=True, path=path, now=MORNING)
    assert key_usage.today(MORNING) == "2026-11-06" and key_usage.today(NEXT_DAY) == "2026-11-07"
    assert key_usage.left("k", ["flash"], {"flash": 20}, path=path, now=NEXT_DAY)[0]["left"] == 20
    assert key_usage.next_reset(MORNING) == datetime(2026, 11, 7, 8, 0, tzinfo=UTC)


def test_public_mode_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDETECTOR_PUBLIC", "on")
    key_usage.record("visitor-key", "flash", path=tmp_path / "usage.json")
    assert not (tmp_path / "usage.json").exists()


def test_every_gemini_reply_counts_and_a_quota_error_empties_the_key_and_teaches_its_limit(monkeypatch):
    replies = iter(["One.", generate_essays.QuotaExhausted(
        "flash is out of quota: Rate limit exceeded for model flash (limit: 20 requests per day on Free Tier).")])

    def answer(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(generate_essays, "_chat", answer)
    assert generate_essays.chat(GEMINI, "k", "flash", "Write.", 0.8) == "One."
    assert key_usage.left("k", ["flash"], {"flash": 20})[0]["left"] == 19
    with pytest.raises(generate_essays.QuotaExhausted):
        generate_essays.chat(GEMINI, "k", "flash", "Write.", 0.8)
    assert key_usage.left("k", ["flash"], {"flash": 20})[0] == {"model": "flash", "used": 1, "limit": 20, "left": 0,
                                                                  "spent": True}
    assert generation_settings.learned_limits() == {"flash": 20}


def test_other_providers_arent_counted(monkeypatch):
    monkeypatch.setattr(generate_essays, "_chat", lambda *args, **kwargs: "Hi.")
    generate_essays.chat("https://api.groq.com/openai/v1", "k", "flash", "Write.", 0.8)
    assert key_usage.left("k", ["flash"], {})[0]["used"] == 0
