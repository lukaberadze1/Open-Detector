"""Batch generation for the red-team generator app, saved block by block.

A batch regenerates every answer box or text block of an uploaded document, which can take an
hour with a local model. Each block is written to a checkpoint file the moment it is generated,
and again once it is scored, so a stop, a crash or a spent quota loses nothing: the next run
finds the checkpoint and resumes from the first block that is missing.

Generation and scoring run as two passes rather than block by block. On an 8 GB machine a local
model and the detector don't fit in memory together, so the local model is unloaded in between.

A BatchWorker runs both passes on its own thread. Run inside a Streamlit script instead, a batch
ends whenever the script reruns: when a widget is touched, or when a background browser tab
reconnects. The page only reads the worker's progress and can ask it to stop.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def batch_key(*parts) -> str:
    """A short stable name for a batch. The same document, mode, instructions and length settings
    give the same key, so a rerun finds its saved progress; changing any of them starts afresh."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Checkpoint:
    """A batch's progress on disk: `meta` describes the batch and `blocks` maps each block's key to
    what is known about it so far (text and model once generated, score once scored)."""

    path: Path
    meta: dict
    blocks: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Checkpoint | None:
        """The saved checkpoint, or None if there is none or it can't be read."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(path, dict(data["meta"]), dict(data["blocks"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @classmethod
    def start(cls, path: Path, meta: dict) -> Checkpoint:
        checkpoint = cls(path, {**meta, "created": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")})
        checkpoint.save()
        return checkpoint

    def save(self) -> None:
        """Write a temporary file and rename it over the old one, so a crash mid-write leaves the
        previous checkpoint whole."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps({"meta": self.meta, "blocks": self.blocks}, ensure_ascii=False),
                             encoding="utf-8")
        os.replace(temporary, self.path)

    def update(self, key: str, **fields) -> None:
        self.blocks.setdefault(key, {}).update(fields)
        self.save()

    def generated(self, keys: Iterable[str]) -> list[str]:
        return [key for key in keys if self.blocks.get(key, {}).get("text")]

    def scored(self, keys: Iterable[str]) -> list[str]:
        return [key for key in self.generated(keys) if "score" in self.blocks[key]]

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def generate_missing(checkpoint: Checkpoint, keys: list[str], generate: Callable[[str], tuple[str, str]],
                     on_block: Callable[[str], None] | None = None) -> None:
    """Generate each block the checkpoint lacks, in order. `generate(key)` returns (text, model); each
    result is saved before `on_block` reports it, so whatever interrupts the batch (an exception from
    `generate`, or Streamlit stopping the script inside `on_block`) loses no finished block."""
    for key in keys:
        if checkpoint.blocks.get(key, {}).get("text"):
            continue
        text, model = generate(key)
        checkpoint.update(key, text=text, model=model)
        if on_block:
            on_block(key)


def score_missing(checkpoint: Checkpoint, keys: list[str], score: Callable[[str], dict | None],
                  on_block: Callable[[str], None] | None = None) -> None:
    """Score each generated block that has no score yet, saving each score as it comes."""
    for key in keys:
        block = checkpoint.blocks.get(key, {})
        if not block.get("text") or "score" in block:
            continue
        checkpoint.update(key, score=score(block["text"]))
        if on_block:
            on_block(key)


class StopRequested(Exception):
    """Raised between blocks once a stop has been asked for."""


class BatchWorker(threading.Thread):
    """Generates, then scores, every missing block of a checkpoint on a thread of its own.

    `phase` moves from "generating" to "scoring" (when there is a `score` function) and ends as
    "done", "stopped" or "failed", with the exception in `error`. `stop()` takes effect once the
    block in progress is saved. `finish`, if given, runs once both passes are complete. `status` is
    a dict the callbacks may fill for the page to show; `seconds` holds how long each block took,
    per pass, for an estimate of the time left."""

    def __init__(self, checkpoint: Checkpoint, keys: list[str], generate: Callable[[str], tuple[str, str]],
                 score: Callable[[str], dict | None] | None = None,
                 before_scoring: Callable[[], None] | None = None, finish: Callable[[], None] | None = None,
                 status: dict | None = None):
        super().__init__(daemon=True)
        self.checkpoint, self.keys = checkpoint, keys
        self._generate, self._score, self._before_scoring, self._finish = generate, score, before_scoring, finish
        self._stop_requested = threading.Event()
        self.status = status if status is not None else {}
        self.phase = "generating"
        self.error: Exception | None = None
        self.seconds: dict[str, list[float]] = {"generated": [], "scored": []}

    def stop(self) -> None:
        self._stop_requested.set()

    @property
    def stopping(self) -> bool:
        return self._stop_requested.is_set()

    def _timed(self, function: Callable, kind: str) -> Callable:
        def timed(argument):
            started = time.monotonic()
            result = function(argument)
            self.seconds[kind].append(time.monotonic() - started)
            return result
        return timed

    def _between_blocks(self, key: str) -> None:
        if self._stop_requested.is_set():
            raise StopRequested

    def run(self) -> None:
        try:
            generate_missing(self.checkpoint, self.keys, self._timed(self._generate, "generated"),
                             self._between_blocks)
            if self._score:
                self.phase = "scoring"
                if self._before_scoring:
                    self._before_scoring()
                score_missing(self.checkpoint, self.keys, self._timed(self._score, "scored"), self._between_blocks)
            if self._finish:
                self._finish()
            self.phase = "done"
        except StopRequested:
            self.phase = "stopped"
        except Exception as exc:  # reported on the page; every finished block is already saved
            self.error, self.phase = exc, "failed"
