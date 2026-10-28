import pytest


@pytest.fixture(autouse=True)
def own_prefs(monkeypatch, tmp_path_factory):
    """Every test remembers the app's choices in a folder of its own, with the Guide already shown, so
    no test reads or writes this computer's data/ui_prefs.json."""
    from ui import prefs

    monkeypatch.setattr(prefs, "PATH", tmp_path_factory.mktemp("prefs") / "ui_prefs.json")
    prefs.update(welcomed=True)
