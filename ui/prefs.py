"""What the app remembers on this computer: the look picked under Settings, and whether the Guide has
been shown. Kept in data/ui_prefs.json in the app's data folder (data/ is gitignored); callers skip
it on a public deploy, which serves strangers."""

from __future__ import annotations

import json

from opendetector.config import DATA_ROOT

PATH = DATA_ROOT / "data" / "ui_prefs.json"


def load() -> dict:
    try:
        prefs = json.loads(PATH.read_text())
    except (OSError, ValueError):
        return {}
    return prefs if isinstance(prefs, dict) else {}


def update(**changes) -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(load() | changes))
    except OSError:
        pass  # a remembered choice is a convenience, never worth an error
