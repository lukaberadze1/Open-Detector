"""Load API keys from the gitignored .env file in the app's data folder, so they never live in the code."""

from __future__ import annotations

import os
from pathlib import Path

from .config import DATA_ROOT

ENV_PATH = DATA_ROOT / ".env"


def _name(line: str) -> str:
    return line.strip().removeprefix("export ").split("=", 1)[0].strip()


def load_env(path: Path | str = ENV_PATH) -> None:
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


def save_env(name: str, value: str, path: Path | str = ENV_PATH) -> None:
    """Set `name` to `value` in `path` and in os.environ, keeping the file's other lines; an empty
    value removes it from both. Only this computer's user can read the file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    kept = [line for line in lines if line.strip().startswith("#") or _name(line) != name]
    if value:
        kept.append(f"{name}={value}")
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def extra_keys_name(name: str) -> str:
    """Where more keys for the same provider are kept, comma-separated: GEMINI_API_KEYS beside GEMINI_API_KEY."""
    return f"{name}S"


def env_keys(name: str) -> list[str]:
    """Every key for `name`'s provider: `name` itself first, then the extra ones, each once."""
    keys = [os.environ.get(name, ""), *os.environ.get(extra_keys_name(name), "").split(",")]
    return list(dict.fromkeys(key.strip() for key in keys if key.strip()))


def save_extra_keys(name: str, keys: list[str], path: Path | str = ENV_PATH) -> None:
    """Keep `keys` as the extra keys beside `name` (empty removes them), never repeating `name`'s own."""
    main = os.environ.get(name, "").strip()
    kept = [key for key in dict.fromkeys(key.strip() for key in keys) if key and key != main]
    save_env(extra_keys_name(name), ",".join(kept), path)
