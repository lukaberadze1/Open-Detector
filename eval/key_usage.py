"""How many Gemini requests each key has made on each model today, counted on this computer.

Gemini's API doesn't say how many requests a key has left, so the app counts its own: every reply a key
gets is one request, and a quota error means the key is out on that model until the daily reset, at
midnight Pacific time. Requests made elsewhere with the same key (AI Studio, another app, another
computer) aren't seen, so what is left is an upper bound until a quota error says otherwise.

Keys are stored only as a short hash, in DATA_ROOT/data/key_usage.json. Public mode records nothing,
since visitors' keys must not be written down in any form. Standard library only, in APP_FILES.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from opendetector.config import DATA_ROOT

PATH = DATA_ROOT / "data/key_usage.json"
_LOCK = threading.Lock()


def _pacific() -> timezone:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:  # no time zone data on this system: Pacific standard time is near enough
        return timezone(timedelta(hours=-8))


def today(now: datetime | None = None) -> str:
    """The quota day: Gemini's daily limits reset at midnight Pacific time."""
    return (now or datetime.now(UTC)).astimezone(_pacific()).date().isoformat()


def next_reset(now: datetime | None = None) -> datetime:
    """When the daily limits next reset, in this computer's own time zone."""
    pacific = (now or datetime.now(UTC)).astimezone(_pacific())
    midnight = datetime.combine(pacific.date() + timedelta(days=1), datetime.min.time(), tzinfo=pacific.tzinfo)
    return midnight.astimezone()


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def enabled() -> bool:
    from eval.gemini_fallback import _public

    return not _public()


def _read(path: Path, day: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("keys", {}) if isinstance(data, dict) and data.get("day") == day else {}


def _write(path: Path, day: str, keys: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spare = path.with_suffix(f".{os.getpid()}.tmp")
    spare.write_text(json.dumps({"day": day, "keys": keys}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spare.replace(path)


def record(key: str, model: str, *, spent: bool = False, path: Path | None = None,
           now: datetime | None = None) -> None:
    """Count one request `key` made on `model` today, or with `spent` note that its quota there ran out.
    Read afresh each time, so the overnight runner and the app add to the same day's counts."""
    if not key or not enabled():
        return
    path, day = path or PATH, today(now)
    with _LOCK:
        keys = _read(path, day)
        entry = keys.setdefault(fingerprint(key), {}).setdefault(model, {"used": 0})
        if spent:
            entry["spent"] = True
        else:
            entry["used"] = entry.get("used", 0) + 1
        with contextlib.suppress(OSError):  # a count is never worth failing a request over
            _write(path, day, keys)


def left(key: str, models: list[str], limits: dict[str, int], path: Path | None = None,
         now: datetime | None = None) -> list[dict]:
    """What `key` has left today on each of `models`: the model, requests used, its daily limit (None when not
    known yet), requests left (None without a limit) and whether a quota error said it is out."""
    counts = _read(path or PATH, today(now)).get(fingerprint(key), {})
    rows = []
    for model in dict.fromkeys(models):
        entry = counts.get(model, {})
        used, spent, limit = entry.get("used", 0), bool(entry.get("spent")), limits.get(model)
        remaining = 0 if spent else (max(0, limit - used) if limit else None)
        rows.append({"model": model, "used": used, "limit": limit, "left": remaining, "spent": spent})
    return rows
