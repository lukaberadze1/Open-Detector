"""Load API keys from the project's gitignored .env file, so they never live in the code."""

from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT


def load_env(path: Path | str = ROOT / ".env") -> None:
    """Copy KEY=VALUE lines from `path` into os.environ. Variables already set win, so an
    exported key overrides the file. A missing file is fine."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if key:
            os.environ.setdefault(key, value.strip().strip("'\""))
