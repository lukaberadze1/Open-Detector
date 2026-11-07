import pytest


@pytest.fixture(autouse=True)
def own_prefs(monkeypatch, tmp_path_factory):
    """Every test remembers the app's choices in a folder of its own, with the Guide already shown, so
    no test reads or writes this computer's data/ui_prefs.json."""
    from ui import prefs

    monkeypatch.setattr(prefs, "PATH", tmp_path_factory.mktemp("prefs") / "ui_prefs.json")
    prefs.update(welcomed=True)


@pytest.fixture(autouse=True)
def own_recent_checks(monkeypatch, tmp_path_factory):
    """Every test keeps Detect's recent checks in a folder of its own, never this computer's data/detect_recent."""
    from ui import detect

    monkeypatch.setattr(detect, "RECENT_DIR", tmp_path_factory.mktemp("recent"))


@pytest.fixture(autouse=True)
def own_limits(monkeypatch, tmp_path_factory):
    """Every test keeps the daily request limits it learns in a file of its own, never this computer's
    data/model_limits.json."""
    from eval import generation_settings

    monkeypatch.setattr(generation_settings, "LIMITS", tmp_path_factory.mktemp("limits") / "model_limits.json")


@pytest.fixture(autouse=True)
def own_key_usage(monkeypatch, tmp_path_factory):
    """Every test counts Gemini requests in a file of its own, never this computer's data/key_usage.json."""
    from eval import key_usage

    monkeypatch.setattr(key_usage, "PATH", tmp_path_factory.mktemp("usage") / "key_usage.json")


@pytest.fixture(autouse=True)
def no_recipe(monkeypatch, tmp_path_factory):
    """Every test starts with no measured recipe, so the shipped models/recipe.json can't change what a page test
    sends; a test that wants one writes it to recipe.TABLE."""
    from eval import recipe

    monkeypatch.setattr(recipe, "TABLE", tmp_path_factory.mktemp("recipe") / "recipe.json")


@pytest.fixture(autouse=True)
def no_gemini_website(monkeypatch):
    """Page tests load the real .env, so without this a failed model would use the owner's Firefox login."""
    monkeypatch.setattr("eval.generate_essays.web_available", lambda: False)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """A run started from a shell that loaded .env (the overnight runner does) mustn't hand the tests its keys: two
    Gemini keys doubled the daily limit test_points_pick_the_batch_settings reads."""
    import os

    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_API_KEYS")) or name.startswith(("GEMINI_SECURE", "GEMINI_WEB")):
            monkeypatch.delenv(name)
