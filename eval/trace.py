"""A record of every step a text takes, so a report can say which step moved the score and what it cost: each version
with its scores by window and sentence, each rewrite that was tried and why it was kept or dropped, and each request
with its route, tokens, thought summary and full prompt. Append-only JSON lines; one writer at a time per file.

It holds whole texts and prompts, style profile included, so it stays in gitignored folders and never in public mode:
only the command-line benchmarks open one.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from eval import generate_essays


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def report_fields(report) -> dict:
    """What the detector said about a version, in the same shape whichever build said it."""
    return {"score": report.ai_probability, "verdict": report.verdict, "words": report.words,
            "features": report.features, "engines": report.engines, "windows": report.windows,
            "sentences": [[s.start, s.end, round(s.ai_probability, 4)] for s in report.sentences]}


class Trace:
    def __init__(self, path: Path, **context) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.head: str | None = None  # the version the next one was made from
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.step = 0

    def bind(self, **context) -> None:
        """Start a new run's context: what follows belongs to it."""
        self.context = {**self.context, **context}
        self.head, self.step = None, 0

    def event(self, kind: str, **fields) -> None:
        with self._lock:
            self.step += 1
            row = {"kind": kind, "at": round(time.time(), 2), "step": self.step, **self.context, **fields}
            with self.path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")

    def version(self, stage: str, text: str, report=None, keep: bool = True, **fields) -> str:
        """A whole text at a stage, made from the last version kept. Returns its id."""
        version = text_id(text)
        self.event("version", stage=stage, version=version, parent=self.head, text=text,
                   **(report_fields(report) if report is not None else {}), **fields)
        if keep:
            self.head = version
        return version

    def _blob(self, text: str | None) -> str | None:
        """A long text stored once, by id: the same style profile goes out with nearly every request."""
        if not text:
            return None
        key = text_id(text)
        if key not in self._seen:
            self._seen.add(key)
            self.event("blob", id=key, text=text)
        return key

    def request(self, **fields) -> None:
        fields["prompt"], fields["system"] = self._blob(fields.get("prompt")), self._blob(fields.get("system"))
        self.event("request", **fields)

    def watch_requests(self) -> None:
        """Every request made from here on goes in this trace."""
        generate_essays.OBSERVER = self.request

    def stop_watching(self) -> None:
        generate_essays.OBSERVER = None
