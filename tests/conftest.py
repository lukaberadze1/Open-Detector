import pytest


@pytest.fixture(autouse=True)
def own_prefs(monkeypatch, tmp_path_factory):
    """Every test remembers the app's choices in a folder of its own, with the Guide already shown, so
    no test reads or writes this computer's data/ui_prefs.json."""
    from ui import prefs

    monkeypatch.setattr(prefs, "PATH", tmp_path_factory.mktemp("prefs") / "ui_prefs.json")
    prefs.update(welcomed=True)


@pytest.fixture(autouse=True)
def own_limits(monkeypatch, tmp_path_factory):
    """Every test keeps the daily request limits it learns in a file of its own, never this computer's
    data/model_limits.json."""
    from eval import generation_settings

    monkeypatch.setattr(generation_settings, "LIMITS", tmp_path_factory.mktemp("limits") / "model_limits.json")
