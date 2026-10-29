"""Batch generation for the red-team generator app, saved block by block.

A batch regenerates every answer box or text block of an uploaded document, which can take an
hour with a local model. Each block is written to a checkpoint file the moment it is generated,
and again once it is scored, so a stop, a crash or a spent quota loses nothing: the next run
finds the checkpoint and resumes from the first block that is missing.

Generation and scoring run as two passes rather than block by block. On an 8 GB machine a local
model and the detector don't fit in memory together, so the local model is unloaded in between.

Blocks are written a group at a time: one request rewrites several neighbouring blocks, so the style
profile and the document it carries are sent once per group instead of once per block, and a few
groups are written at the same time. `parse_group_reply` reads a group's JSON reply.

The prompts are here too, so the Red team page and the settings benchmark (eval/tune_generation.py)
send the same ones. They take the page's plan: anything with `prompt`, `context_text`, `blocks`,
`answer_slots`, `match_existing_lengths` and `answer_words`.

A BatchWorker runs both passes on its own thread. Run inside a Streamlit script instead, a batch
ends whenever the script reruns: when a widget is touched, or when a background browser tab
reconnects. The page only reads the worker's progress and can ask it to stop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from datetime import UTC, datetime
from pathlib import Path

from opendetector.segment import count_words


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
    # Blocks written at the same time are saved from several threads.
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

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
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps({"meta": self.meta, "blocks": self.blocks}, ensure_ascii=False),
                                 encoding="utf-8")
            os.replace(temporary, self.path)

    def update(self, key: str, **fields) -> None:
        with self._lock:
            self.blocks.setdefault(key, {}).update(fields)
            self.save()

    def generated(self, keys: Iterable[str]) -> list[str]:
        return [key for key in keys if self.blocks.get(key, {}).get("text")]

    def first_missing(self, keys: list[str]) -> int | None:
        """The position, counted from 1, of the first block not yet generated. Blocks written at the
        same time can finish out of order, so this can come before blocks that are saved."""
        return next((index for index, key in enumerate(keys, 1) if not self.blocks.get(key, {}).get("text")), None)

    def scored(self, keys: Iterable[str]) -> list[str]:
        return [key for key in self.generated(keys) if "score" in self.blocks[key]]

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def generate_missing(checkpoint: Checkpoint, keys: list[str],
                     generate: Callable[[list[str]], Iterable[tuple[str, str, str]]],
                     on_block: Callable[[str], None] | None = None, group_size: int = 1, workers: int = 1) -> None:
    """Generate each block the checkpoint lacks, in groups of up to `group_size` neighbouring blocks,
    `workers` groups at a time. `generate(group)` yields (key, text, model) for each block as it is
    written, and each is saved before `on_block` reports it. Whatever interrupts the batch (an exception
    from `generate`, or from `on_block` when a stop is asked for) starts no new group, waits for the
    groups being written, and is raised once they are saved, so no finished block is lost."""
    missing = [key for key in keys if not checkpoint.blocks.get(key, {}).get("text")]
    size = max(1, group_size)
    groups = iter([missing[start:start + size] for start in range(0, len(missing), size)])

    def write(group: list[str]) -> tuple[list[str], BaseException | None]:
        saved = []
        try:
            for key, text, model in generate(group):
                checkpoint.update(key, text=text, model=model)
                saved.append(key)
        except BaseException as exc:  # raised on the batch's own thread, after the groups in progress
            return saved, exc
        return saved, None

    interrupted: BaseException | None = None
    with ThreadPoolExecutor(max(1, workers)) as pool:
        running = {pool.submit(write, group) for group in islice(groups, max(1, workers))}
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                saved, error = future.result()
                for key in saved:
                    if on_block and interrupted is None:
                        try:
                            on_block(key)
                        except BaseException as exc:
                            interrupted = exc
                interrupted = interrupted or error
            if interrupted is None:
                running |= {pool.submit(write, group) for group in islice(groups, len(finished))}
    if interrupted is not None:
        raise interrupted


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


# -- prompts --------------------------------------------------------------------------


def person_instruction(*samples: str) -> str:
    """Infer grammatical person from existing answer prose, ignoring question wording where possible."""
    text = " ".join(samples).casefold()
    counts = {
        "first person": len(re.findall(r"\b(?:i|me|my|mine|we|us|our|ours)\b", text)),
        "second person": len(re.findall(r"\b(?:you|your|yours|yourself|yourselves)\b", text)),
        "third person": len(re.findall(r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b", text)),
    }
    person, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        return (
            "Match the grammatical person used in the source text; do not add first- or "
            "second-person language if it is absent."
        )
    return f"Write in {person}, matching the source text. Keep that grammatical person consistent throughout."


def target_words(plan, job: dict) -> int:
    """About how long a block's new text should be."""
    if plan.match_existing_lengths and job["previous"].strip():
        return count_words(job["previous"])
    return plan.answer_words


def voice_rule(plan, jobs: list[dict]) -> str:
    """Which grammatical person blocks are written in, read from their own text and the rest of the
    document's answers or text blocks."""
    samples = [job["previous"] for job in jobs]
    if any(job["kind"] == "answer_box" for job in jobs):
        samples.extend(slot.previous_answer for slot in plan.answer_slots if slot.previous_answer)
    if any(job["kind"] != "answer_box" for job in jobs):
        samples.extend(section.text for section in plan.blocks if section.text not in samples[:len(jobs)])
    return person_instruction(*samples)


def group_prompt(plan, jobs: list[dict]) -> str:
    """The prompt that writes several neighbouring blocks in one request, replying in JSON. It starts
    with the same document as every other request of the batch, so the server can reuse what it read,
    and gives each block an id (B1, B2, …) for the reply."""
    items = []
    for number, job in enumerate(jobs, 1):
        aim = f"Aim for about {target_words(plan, job)} words."
        if job["kind"] == "answer_box":
            items.append(f"[B{number}] A new answer for the question in {job['label']}. {aim}\n"
                         f"Question to answer:\n{job['question']}\n"
                         f"Previous answer in this right-hand box (may be blank):\n{job['previous'] or '(blank)'}")
        else:
            kind = "Markdown text" if job["kind"] == "markdown_block" else "text"
            items.append(f"[B{number}] A rewrite of the existing {kind} block labelled {job['label']}. {aim}\n"
                         f"Existing text to regenerate:\n{job['previous']}")
    return (
        f"Full document context:\n{plan.context_text}\n\n---\n\n"
        f"Write new text for each of the {len(jobs)} blocks below, which are parts of the document above. "
        f"{voice_rule(plan, jobs)} "
        "Keep each block's meaning and role in the document, using the surrounding document as context. "
        "Write each block on its own: don't merge blocks, move text between them, or add headings, questions, "
        "table rows or any block not listed. A new answer replaces the previous one, if there is one, and "
        "answers only its own question. These blocks are AI-generated red-team evaluation data.\n\n"
        f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
        'Reply with JSON only, in the form {"blocks": [{"id": "B1", "text": "..."}]}: one entry for every '
        "block, in the order given, holding only that block's new text.\n\n"
        "Blocks:\n\n" + "\n\n".join(items)
    )


def block_prompt(plan, job: dict) -> str:
    """The prompt that writes one block of a batch. The document comes first and is the same for every
    block, so a local model can keep it cached and read only the block's own part each time."""
    voice = voice_rule(plan, [job])
    words = target_words(plan, job)
    if job["kind"] == "answer_box":
        generation_prompt = (
            f"Other document context:\n{plan.context_text}\n\n---\n\n"
            f"Generate a new AI red-team answer for the question in {job['label']}. "
            f"Aim for about {words} words. "
            f"{voice} "
            "Use the document's other questions and previous answers as context. "
            "Replace the existing answer if one "
            "is present. Do not change, repeat, or answer any other question. Return only the new answer text. "
            "This answer is AI-generated red-team evaluation data.\n\n"
            f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
            f"Question to answer:\n{job['question']}\n\n"
            f"Previous answer in this right-hand box (may be blank):\n{job['previous'] or '(blank)'}"
        )
    else:
        generation_prompt = (
            f"Full document context:\n{plan.context_text}\n\n---\n\n"
            f"Rewrite only the existing {'Markdown text' if job['kind'] == 'markdown_block' else 'text'} block "
            f"labelled {job['label']}. "
            f"Aim for about {words} words. "
            f"{voice} "
            "Keep its meaning and role in the document, using the surrounding document as context. "
            "Return only the "
            "replacement for this block. Do not output headings, questions, table rows, or any other block. "
            "This replacement is AI-generated red-team evaluation data.\n\n"
            f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
            f"Existing text to regenerate:\n{job['previous']}"
        )
    return generation_prompt


# -- running a batch -------------------------------------------------------------------


class StopRequested(Exception):
    """Raised between blocks once a stop has been asked for."""


# A group's reply: the new text of each block, by the id the prompt gave it.
GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                      "required": ["id", "text"]},
        },
    },
    "required": ["blocks"],
}


def parse_group_reply(reply: str, ids: list[str]) -> dict[str, str]:
    """The text a group's reply gives each block, by id. A model asked for JSON without a schema may
    wrap it in a code fence or a sentence, so the outermost braces are read. Ids it doesn't know, gives
    twice or leaves empty are left out, and those blocks are written again one at a time."""
    start, end = reply.find("{"), reply.rfind("}")
    try:
        data = json.loads(reply[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        return {}
    entries = data.get("blocks") if isinstance(data, dict) else None
    texts: dict[str, str] = {}
    repeated = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            continue
        block_id = str(entry.get("id", "")).strip()
        if block_id in texts:
            repeated.add(block_id)
        if block_id in ids and entry["text"].strip():
            texts[block_id] = entry["text"].strip()
    return {block_id: text for block_id, text in texts.items() if block_id not in repeated}


class BatchWorker(threading.Thread):
    """Generates, then scores, every missing block of a checkpoint on a thread of its own.

    `phase` moves from "generating" to "scoring" (when there is a `score` function) and ends as
    "done", "stopped" or "failed", with the exception in `error`. `stop()` takes effect once the
    groups in progress are saved. `finish`, if given, runs once both passes are complete, and `on_end`
    runs last however the batch ended. `status` is a dict the callbacks may fill for the page to show.
    `seconds` holds the time between finished blocks, per pass, and `window` how many of the latest
    the page averages for the time left: enough to cover a few rounds of groups."""

    def __init__(self, checkpoint: Checkpoint, keys: list[str],
                 generate: Callable[[list[str]], Iterable[tuple[str, str, str]]],
                 score: Callable[[str], dict | None] | None = None,
                 before_scoring: Callable[[], None] | None = None, finish: Callable[[], None] | None = None,
                 status: dict | None = None, on_end: Callable[[], None] | None = None,
                 group_size: int = 1, workers: int = 1):
        super().__init__(daemon=True)
        self.checkpoint, self.keys = checkpoint, keys
        self._generate, self._score, self._before_scoring, self._finish = generate, score, before_scoring, finish
        self._on_end = on_end
        self.group_size, self.workers = max(1, group_size), max(1, workers)
        self._stop_requested = threading.Event()
        self.status = status if status is not None else {}
        self.phase = "generating"
        self.error: Exception | None = None
        self.seconds: dict[str, list[float]] = {"generated": [], "scored": []}
        self.window = max(8, 2 * self.group_size * self.workers)

    def stop(self) -> None:
        self._stop_requested.set()

    @property
    def stopping(self) -> bool:
        return self._stop_requested.is_set()

    def _reporter(self, kind: str) -> Callable[[str], None]:
        """Called as each block is saved: records the time since the last one, then stops the batch
        if a stop has been asked for. Blocks written at the same time finish close together, so these
        times are a block's share of the wall clock, not how long its request took."""
        last = time.monotonic()

        def report(key: str) -> None:
            nonlocal last
            now = time.monotonic()
            self.seconds[kind].append(now - last)
            last = now
            if self._stop_requested.is_set():
                raise StopRequested
        return report

    def run(self) -> None:
        try:
            generate_missing(self.checkpoint, self.keys, self._generate, self._reporter("generated"),
                             self.group_size, self.workers)
            if self._score:
                self.phase = "scoring"
                if self._before_scoring:
                    self._before_scoring()
                score_missing(self.checkpoint, self.keys, self._score, self._reporter("scored"))
            if self._finish:
                self._finish()
            self.phase = "done"
        except StopRequested:
            self.phase = "stopped"
        except Exception as exc:  # reported on the page; every finished block is already saved
            self.error, self.phase = exc, "failed"
        finally:
            if self._on_end:
                self._on_end()
