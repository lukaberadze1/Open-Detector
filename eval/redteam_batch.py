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
`answer_slots`, `match_existing_lengths` and `answer_words`, and optionally `structure`: how closely
a rewritten text block follows the original (STRUCTURES; free when absent).

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
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice, pairwise
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from opendetector.segment import Sentence, count_words, split_sentences


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


# A block of about this many words takes one place in a group. The settings benchmark's blocks averaged
# about this length, so "8 blocks per request" means the same request size there and here.
UNIT_WORDS = 30


def group_by_words(keys: list[str], size: int, words: Callable[[str], int] | None = None) -> list[list[str]]:
    """Neighbouring keys in groups that fill up to `size` places. With `words`, a block takes a place
    per UNIT_WORDS words it should come to, and at least a quarter of one: a run of one-line blocks
    shares a request, and a long paragraph takes the room of several. Size 1 is a block per request."""
    size = max(1, size)
    if size == 1 or words is None:
        return [keys[start:start + size] for start in range(0, len(keys), size)]
    groups: list[list[str]] = []
    filled = 0.0
    for key in keys:
        place = max(0.25, words(key) / UNIT_WORDS)
        if groups and filled + place <= size:
            groups[-1].append(key)
            filled += place
        else:
            groups.append([key])
            filled = place
    return groups


def generate_missing(checkpoint: Checkpoint, keys: list[str],
                     generate: Callable[[list[str]], Iterable[tuple[str, str, str]]],
                     on_block: Callable[[str], None] | None = None, group_size: int = 1, workers: int = 1,
                     words: Callable[[str], int] | None = None) -> None:
    """Generate each block the checkpoint lacks, in groups of neighbouring blocks (`group_by_words`),
    `workers` groups at a time. `generate(group)` yields (key, text, model) for each block as it is
    written, and each is saved before `on_block` reports it. Whatever interrupts the batch (an exception
    from `generate`, or from `on_block` when a stop is asked for) starts no new group, waits for the
    groups being written, and is raised once they are saved, so no finished block is lost."""
    missing = [key for key in keys if not checkpoint.blocks.get(key, {}).get("text")]
    groups = iter(group_by_words(missing, group_size, words))

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


def worth_rewriting(text: str) -> bool:
    """Whether a text block is prose worth rewriting: a sentence of four words or more, or eight words
    without one. A table cell's label, a name, a date or a number ("14 August", "Done", "1") stays as
    it is: rewriting it costs a request's share, adds a sample the detector can't learn from, and can
    break the document."""
    words = len(re.findall(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", text))  # words of letters, not "1" or "."
    return words >= 8 or (words >= 4 and bool(re.search(r"[.!?]", text)))


def person_instruction(*samples: str) -> str:
    """Infer grammatical person from existing answer prose, ignoring question wording where possible.
    Nothing without any: a sample written from a prompt alone was once told not to add first- or
    second-person language "if it is absent" from a source it didn't have, which kept every "I" and
    "you" out of it, and people write both (eval/build_human_profile.py)."""
    text = " ".join(samples).casefold()
    if not text.strip():
        return ""
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


# -- following the original --------------------------------------------------------------

# How closely a rewritten text block follows the original, whoever wrote it. "same" hides the new text
# in the original's skeleton: its sentences one for one, their lengths, punctuation and layout. That
# hides it in a human's skeleton but not in an AI's, whose even sentences survive any rewrite in place,
# so "writer" gives each block a skeleton borrowed from a passage of the style profile's own writer.
STRUCTURES = {"light": "Light edit", "same": "Same structure", "writer": "Writer's sentences",
              "free": "Free rewrite"}
FOLLOW = {
    "writer": "Write it in the sentence plan given with it, taken from a passage of the writer's own: one "
              "sentence for each numbered line, in that order, each within a few words of its length and with "
              "the punctuation listed. Fit the original's meaning into those sentences, merging, splitting or "
              "reordering its points as the plan needs, in new words. Leave out the numbers.",
    "light": "Edit the original lightly, sentence by sentence: keep every numbered sentence in its order, with "
             "its length, its punctuation and most of its words, and change about one word or phrase in four. "
             "Keep its casing, spacing and spelling as they are.",
    "same": "Follow the original sentence by sentence: write one new sentence for each numbered sentence, in the "
            "same order, within a few words of its length and with the same punctuation (its commas, brackets, "
            "dashes and questions). Say the same thing in new words. Keep the original's casing, spacing and "
            "spelling habits. Leave out the numbers and word counts.",
    # Told to write like the writer of an original, a model otherwise hands much of it back unchanged.
    "free": "Say the same thing in new words: never hand back the original's sentences.",
}
_MARKS = re.compile(r"[,;:()\[\]\u2013\u2014?!\"\u201c\u201d]|\s-\s")
_WORDS = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")
_NUMBERED = re.compile(r"^[ \t]*\d+\.[ \t]+(?:\(\d+ words?\)[ \t]*)?", re.M)


MARK_NAMES = ((",", "comma"), ("(", "bracket pair"), (";", "semicolon"), (":", "colon"), ("?", "question mark"))


def structure(plan) -> str:
    """How closely the plan's rewrites follow their originals: "writer" without the writer's sentence
    shapes to borrow (no profile, or a visitor's own on a public deploy) works as "same"."""
    chosen = getattr(plan, "structure", "free")
    return "same" if chosen == "writer" and not getattr(plan, "shapes", None) else chosen


def sentence_shape(sentence: str) -> tuple[int, str]:
    """A sentence's length in words and its punctuation, the way a sentence plan lists them."""
    marks = [f"{count} {name}{'s' if count > 1 else ''}" for mark, name in MARK_NAMES
             if (count := sentence.count(mark))]
    return count_words(sentence), ", ".join(marks)


def paragraph_shapes(paragraphs: Iterable[str]) -> list[list[tuple[int, str]]]:
    """Each paragraph's sentences as shapes (sentence_shape), the skeletons "writer" borrows from."""
    return [[sentence_shape(text[s.start:s.end]) for s in split_sentences(text)] for text in paragraphs]


def borrowed_skeleton(shapes: list[list[tuple[int, str]]], words: int, key: str) -> list[tuple[int, str]] | None:
    """A run of consecutive sentences from one of the writer's paragraphs, `words` long in all to within a
    quarter, picked by the block's `key` so a block keeps its plan when it is written again. None when
    nothing the writer wrote fits."""
    fits = []
    for paragraph in shapes:
        for start in range(len(paragraph)):
            total = 0
            for end in range(start, len(paragraph)):
                total += paragraph[end][0]
                if total > 1.25 * words:
                    break
                if total >= 0.75 * words:
                    fits.append(paragraph[start:end + 1])
    return fits[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(fits)] if fits else None


def original_text(plan, job: dict) -> str:
    """A text block's original as the prompt shows it. When the rewrite follows it, its sentences are
    numbered with their lengths, a blank line between paragraphs, so the model can match each one. In
    "writer" it comes as it is, with the sentence plan borrowed from the writer (plan.shapes), or the
    original's own when the writer has nothing that fits."""
    if structure(plan) == "free" or job["kind"] == "answer_box":
        return job["previous"]
    if structure(plan) == "writer":
        words = round(count_words(job["previous"]) * getattr(plan, "stretch", 1.0))
        skeleton = (borrowed_skeleton(getattr(plan, "shapes", None) or [], words, job.get("key", job["label"]))
                    or paragraph_shapes([job["previous"]])[0])
        lines = [f"{number}. {length} word{'s' if length != 1 else ''}" + (f", {marks}" if marks else "")
                 for number, (length, marks) in enumerate(skeleton, 1)]
        return job["previous"] + "\n\nSentence plan:\n" + "\n".join(lines)
    sentences, lines = split_sentences(job["previous"]), []
    for number, sentence in enumerate(sentences, 1):
        if number > 1 and sentence.paragraph != sentences[number - 2].paragraph:
            lines.append("")
        lines.append(f"{number}. ({sentence.words} words) {job['previous'][sentence.start:sentence.end]}")
    return "\n".join(lines) or job["previous"]


def follow_original(plan, job: dict, text: str) -> str:
    """A rewritten text block laid out like the original: with as many sentences, the original's
    paragraph breaks, line breaks and spacing between sentences go around the new ones, so the rewrite
    keeps the original writer's layout, double spaces included. Otherwise the rewrite as it came, on the
    original's lines."""
    if structure(plan) == "free" or job["kind"] == "answer_box" or not text.strip():
        return text
    original = job["previous"]
    text = _NUMBERED.sub("", text).strip()
    old, new = split_sentences(original), split_sentences(text)
    if structure(plan) == "writer" or not old or len(old) != len(new):  # not in the original's sentences
        return text if "\n" in original else re.sub(r"[ \t]*\n[ \t\n]*", " ", text)
    parts = [original[:old[0].start]]
    for index, (was, now) in enumerate(zip(old, new, strict=True)):
        parts.append(text[now.start:now.end])
        parts.append(original[was.end:old[index + 1].start] if index + 1 < len(old) else original[was.end:])
    return "".join(parts)


def _similarity(first: list[int], second: list[int]) -> float:
    """1 minus the summed differences over the summed larger values, position by position; a position
    one side lacks counts in full."""
    size = max(len(first), len(second))
    first, second = first + [0] * (size - len(first)), second + [0] * (size - len(second))
    total = sum(max(a, b) for a, b in zip(first, second, strict=True))
    return 1.0 if total == 0 else 1 - sum(abs(a - b) for a, b in zip(first, second, strict=True)) / total


def _paragraph_sizes(sentences: list[Sentence]) -> list[int]:
    counts = Counter(sentence.paragraph for sentence in sentences)
    return [counts[index] for index in sorted(counts)]


def rewrite_measures(original: str, rewrite: str) -> dict:
    """How a rewrite sits on its original, for a saved sample. `structure_match` (0 to 100) is the mean
    of three similarities taken sentence by sentence in order: their lengths in words, their punctuation
    marks, and how many sentences each paragraph holds. `words_kept` is the share of the original's words
    the rewrite still has (as a multiset): how much of the original writer's wording survives."""
    old, new = split_sentences(original), split_sentences(rewrite)
    if not old or not new:
        return {}
    marks = [[len(_MARKS.findall(text[s.start:s.end])) for s in sentences]
             for text, sentences in ((original, old), (rewrite, new))]
    parts = (_similarity([s.words for s in old], [s.words for s in new]), _similarity(*marks),
             _similarity(_paragraph_sizes(old), _paragraph_sizes(new)))
    before = Counter(word.casefold() for word in _WORDS.findall(original))
    after = Counter(word.casefold() for word in _WORDS.findall(rewrite))
    return {"structure_match": round(100 * sum(parts) / len(parts), 1),
            "words_kept": round(100 * sum((before & after).values()) / max(1, sum(before.values())), 1)}


# A rewrite keeping this share of the original's words or more is still the original. Told to write
# like the writer of an original, models often hand it back: 24 to 31 of 32 paragraphs word for word
# on 2026-10-30. Saved as AI text, the writer's own words would teach the detector they are AI.
COPIED = 90
AGAIN = "Your last answer repeated the original almost word for word. Write it again: the same meaning in new words."
AGAIN_ALL = ("Your last answer repeated these originals almost word for word. Write each block again: the same "
             "meaning in new words.")


def copied(original: str, rewrite: str) -> bool:
    """Whether a rewrite of `original` is still the original (COPIED% or more of its words)."""
    return rewrite_measures(original, rewrite).get("words_kept", 0) >= COPIED


def rewrite_group(plan, group: list[dict], ask: Callable[[str, list[dict]], str],
                  clean: Callable[[str], str]) -> tuple[dict[str, str], list[dict]]:
    """Write a group of blocks in one request (`ask(prompt, jobs)` gives the reply), then the blocks it
    handed back as their originals once more, together and told so (AGAIN_ALL): at most two requests,
    where asking for each copy on its own would send the profile and the document again per block. Gives
    each written block's text by key, a copy included if it comes back twice (it is never saved as AI),
    and the blocks a reply left out, to be written on their own."""
    texts, alone, pending = {}, [], group
    for attempt in range(2):
        prompt = group_prompt(plan, pending) + (f"\n\n{AGAIN_ALL}" if attempt else "")
        parsed = parse_group_reply(ask(prompt, pending), [f"B{number}" for number in range(1, len(pending) + 1)])
        copies = []
        for number, job in enumerate(pending, 1):
            if not (text := follow_original(plan, job, clean(parsed.get(f"B{number}", "")))):
                alone.append(job)
                continue
            texts[job["key"]] = text
            if copied(job["previous"], text):
                copies.append(job)
        if not copies:
            break
        pending = copies
    return texts, alone


def copied_row(row: dict) -> bool:
    """Whether a saved AI-labelled row is a copy of what it was meant to rewrite: by the words_kept it
    was saved with, or by the original the Red team page saves in its prompt."""
    if row.get("label") not in (1, "1", "ai"):
        return False
    if isinstance(row.get("words_kept"), int | float):
        return row["words_kept"] >= COPIED
    match = re.search(r"Previous text:\n(.*?)\n\nAdditional instructions:", row.get("prompt") or "", re.S)
    return bool(match and match.group(1).strip()) and copied(match.group(1), row.get("text") or "")


def borrowed_plan(shapes: list[list[tuple[int, str]]], words: int, key: str) -> list[list[tuple[int, str]]] | None:
    """A whole piece's sentence plan borrowed from the writer: consecutive paragraphs from one place in
    their writing, cut at the sentence that brings it to `words` words, picked by `key`. None when the
    writer has nothing that long."""
    starts = [index for index in range(len(shapes))
              if sum(length for paragraph in shapes[index:] for length, _ in paragraph) >= words]
    if not starts:
        return None
    plan, total = [], 0
    for paragraph in shapes[starts[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(starts)]:]:
        plan.append([])
        for shape in paragraph:
            plan[-1].append(shape)
            total += shape[0]
            if total >= words:
                return plan
    return plan


# Fitting a draft's points into borrowed sentences, models wrote 50-66% of the draft's words on 2026-10-30
# (samples fell under the detector's 150 words): a restyled passage borrows a plan this much longer, and
# a draft to be restyled is asked for this much longer than the sample.
RESTYLE_STRETCH = 1.35
DRAFT_STRETCH = 1.25


_PARAGRAPH = re.compile(r"\S(?:(?!\n[ \t]*\n).)*", re.S)


def passage_spans(text: str, words: int = 80) -> list[tuple[int, int]]:
    """Where a text's passages are: its paragraphs, a long one cut at sentence ends into passages of up
    to about `words` words, each short enough for a skeleton borrowed from the writer to fit."""
    spans = []
    for paragraph in _PARAGRAPH.finditer(text):
        base, body = paragraph.start(), paragraph.group().rstrip()
        start, count = 0, 0
        for sentence in split_sentences(body):
            count += sentence.words
            if count >= words:
                spans.append((base + start, base + sentence.end))
                start, count = sentence.end, 0
        if body[start:].strip():
            spans.append((base + start, base + len(body)))
    return [(start + len(text[start:end]) - len(text[start:end].lstrip()), end) for start, end in spans]


def draft_jobs(text: str, words: int = 80) -> list[dict]:
    """A draft as text blocks to rewrite: its passages (passage_spans)."""
    return [{"kind": "text_block", "label": f"Passage {number}", "question": "", "previous": text[start:end],
             "key": f"{number:03d} draft"} for number, (start, end) in enumerate(passage_spans(text, words), 1)]


def restyle(draft: str, shapes: list, ask: Callable[[str, list[dict]], str], clean: Callable[[str], str]) -> str:
    """A draft rewritten passage by passage in the writer's sentences (the "writer" structure), as a
    document rewrite does, the passages joined back into the draft's paragraphs. Written from a prompt
    alone, a sample keeps the model's own plan of what to say, which a detector reads whatever voice
    it is told to write in; rewriting a draft in borrowed skeletons breaks that plan up the way the
    document rewrites that get past the detector do. A passage the reply leaves out stays as drafted."""
    jobs = draft_jobs(draft)
    if not jobs:
        return draft
    plan = SimpleNamespace(prompt="", context_text=draft, blocks=[], answer_slots=[], match_existing_lengths=True,
                           answer_words=100, structure="writer", shapes=shapes, stretch=RESTYLE_STRETCH)
    written, _ = rewrite_group(plan, jobs, ask, clean)
    paragraphs, passage = [], 0
    for part in (part.strip() for part in re.split(r"\n\s*\n", draft) if part.strip()):
        pieces = []
        while passage < len(jobs) and jobs[passage]["previous"] in part:
            pieces.append(written.get(jobs[passage]["key"]) or jobs[passage]["previous"])
            passage += 1
        paragraphs.append(" ".join(pieces) or part)
    return "\n\n".join(paragraphs)


# Told what the detector flagged, in the same group request as any other rewrite.
REVISE = ("A detector reads the passages below as machine-written. Write each again the way the writer would "
          "really put it, not as a tidy summary: keep its point, but trade a general statement for one concrete, "
          "particular detail, use the writer's plain, specific words over safe, common ones, and leave out "
          "signposting, wrap-up lines and balanced pairs or lists.")


def passage_scores(spans: list[tuple[int, int]], sentences: list) -> list[float]:
    """Each passage's mean sentence AI likelihood from a detector Report's `sentences` (0 with none)."""
    scores = []
    for start, end in spans:
        inside = [s.ai_probability for s in sentences if start <= s.start < end]
        scores.append(sum(inside) / len(inside) if inside else 0.0)
    return scores


def revise_flagged(text: str, analyze: Callable[[str], object], ask: Callable[[str, list[dict]], str],
                   clean: Callable[[str], str], shapes: list, rounds: int = 2,
                   target: float = 0.2) -> tuple[str, list[float | None]]:
    """A text revised where the detector reads it as AI: each round rewrites its most AI-like passages
    (by sentence score, at most half of them) in new sentence plans borrowed from the writer, told what
    was flagged (REVISE), and keeps each rewritten passage only if the whole text then scores lower,
    trying them most flagged first. Kept or dropped as a whole, a round lost every good passage to one
    bad one, and the samples that stayed flagged were the ones whose rounds were dropped. Stops once it
    scores under `target`. Rewriting a whole draft again keeps the model's own plan of what to say;
    rewriting only where the detector reads it breaks that plan up where it shows. `analyze` is the
    detector's (with sentence scores), run once per rewritten passage. Gives the kept text, normalized as
    the detector read it, and its AI likelihood after each round, the first before any."""
    report = analyze(text)
    if report.ai_probability is None:
        return text, [None]
    best, history = report, [report.ai_probability]
    for round_ in range(1, rounds + 1):
        if best.ai_probability < target:
            break
        base, spans = best.text, passage_spans(best.text)
        if not spans:
            break
        scores = passage_scores(spans, best.sentences)
        order = sorted(range(len(spans)), key=lambda index: -scores[index])
        chosen = [index for index in order if scores[index] >= target][:max(1, len(spans) // 2)] or order[:1]
        jobs = [{"kind": "text_block", "label": f"Passage {index + 1}", "question": "",
                 "previous": base[slice(*spans[index])], "key": f"{index + 1:03d} revise {round_}"}
                for index in sorted(chosen)]
        plan = SimpleNamespace(prompt=REVISE, context_text=base, blocks=[], answer_slots=[],
                               match_existing_lengths=True, answer_words=100, structure="writer" if shapes else "free",
                               shapes=shapes)  # no stretch: a revised passage keeps its length, round after round
        written, _ = rewrite_group(plan, jobs, ask, clean)
        passages = [base[start:end] for start, end in spans]
        gaps = [base[:spans[0][0]]] + [base[end:after[0]] for (_, end), after in pairwise(spans)] \
            + [base[spans[-1][1]:]]
        for index in chosen:
            new = written.get(f"{index + 1:03d} revise {round_}", "").strip()
            if not new or new == passages[index]:
                continue
            trial = [*passages[:index], new, *passages[index + 1:]]
            candidate = analyze("".join(gap + passage for gap, passage in zip(gaps, [*trial, ""], strict=True)))
            if candidate.ai_probability is not None and candidate.ai_probability < best.ai_probability:
                best, passages = candidate, trial
        history.append(best.ai_probability)
    return best.text, history


# How a standard sample is asked for. "challenge" names it an adversarial challenge sample, as the page
# first did; "task" asks for the piece itself, as a writer would be asked. Both are saved as AI text.
FRAMINGS = ("task", "challenge")


def sample_prompt(material: str, words: int, voice: str, framing: str = "task",
                  plan: list[list[tuple[int, str]]] | None = None) -> str:
    """The prompt for a new sample written from `material` (the instructions and any source text). With
    a `plan` (borrowed_plan) it is written in those paragraphs and sentence shapes."""
    voice = f"{voice} " if voice else ""
    if framing == "challenge":
        opening = ("Write an AI-generated adversarial challenge sample based on the test material below. "
                   f"Aim for about {words} words. {voice}Return the sample only. This sample will be labelled "
                   "as AI-generated evaluation data.")
    else:
        opening = (f"Write the piece described below, about {words} words. {voice}Write it the way the writer "
                   "would on their own: their level of detail, their own points in their own order, not a "
                   "complete, balanced overview. Return only the text.")
    if plan:
        lines, number = [], 0
        for paragraph in plan:
            lines.append("")
            for length, marks in paragraph:
                number += 1
                lines.append(f"{number}. {length} word{'s' if length != 1 else ''}" + (f", {marks}" if marks else ""))
        opening += ("\n\nWrite it in this sentence plan, taken from a passage of the writer's own: one sentence for "
                    "each numbered line, in that order, each within a few words of its length and with the "
                    "punctuation listed, a new paragraph at each blank line. Leave out the numbers.\n"
                    + "\n".join(lines))
    return f"{opening}\n\nMaterial:\n{material}" if framing == "task" else f"{opening}\n\nTest material:\n{material}"


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
                         f"Existing text to regenerate:\n{original_text(plan, job)}")
    follow = FOLLOW.get(structure(plan)) if any(job["kind"] != "answer_box" for job in jobs) else None
    return (
        f"Full document context:\n{plan.context_text}\n\n---\n\n"
        f"Write new text for each of the {len(jobs)} blocks below, which are parts of the document above. "
        f"{voice_rule(plan, jobs)} "
        "Keep each block's meaning and role in the document, using the surrounding document as context. "
        + (f"In each rewrite, {follow[0].lower()}{follow[1:]} " if follow else "") +
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
            + (f"{FOLLOW[structure(plan)]} " if structure(plan) in FOLLOW else "") +
            "Return only the "
            "replacement for this block. Do not output headings, questions, table rows, or any other block. "
            "This replacement is AI-generated red-team evaluation data.\n\n"
            f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
            f"Existing text to regenerate:\n{original_text(plan, job)}"
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
    the page averages for the time left: enough to cover a few rounds of groups. `words(key)` is how
    long a block should come out, for grouping by words."""

    def __init__(self, checkpoint: Checkpoint, keys: list[str],
                 generate: Callable[[list[str]], Iterable[tuple[str, str, str]]],
                 score: Callable[[str], dict | None] | None = None,
                 before_scoring: Callable[[], None] | None = None, finish: Callable[[], None] | None = None,
                 status: dict | None = None, on_end: Callable[[], None] | None = None,
                 group_size: int = 1, workers: int = 1, words: Callable[[str], int] | None = None):
        super().__init__(daemon=True)
        self.checkpoint, self.keys = checkpoint, keys
        self._generate, self._score, self._before_scoring, self._finish = generate, score, before_scoring, finish
        self._on_end = on_end
        self.group_size, self.workers, self._words = max(1, group_size), max(1, workers), words
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
                             self.group_size, self.workers, self._words)
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
            # The writers hold the connection, a public visitor's own API key included, and an ended batch
            # stays listed until its page comes back: let them go with the batch.
            self._generate = self._score = self._before_scoring = self._finish = self._on_end = None
