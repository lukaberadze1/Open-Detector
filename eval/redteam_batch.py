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

from eval.fact_mask import LOST, FactMask
from opendetector.normalize import normalize
from opendetector.segment import Sentence, count_words, sentence_windows, split_sentences


def batch_key(*parts) -> str:
    """A short stable name for a batch. The same document, mode, instructions and length settings
    give the same key, so a rerun finds its saved progress; changing any of them starts afresh."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def replace_file(source: Path, target: Path, tries: int = 20) -> None:
    """os.replace that waits out a moment's lock. Windows refuses while anything has the target open: the
    page reading the batch's progress, or antivirus looking at the file just written."""
    for attempt in range(tries):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


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
            replace_file(temporary, self.path)

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
              "reordering its points as the plan needs, in new words. Every sentence is a whole sentence that "
              "reads naturally, and a long line is one long sentence, its clauses joined by commas and 'and'; "
              "when a point won't fit, go over the length rather than cut a sentence off or leave a lone word. "
              "Leave out the numbers.",
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


MIN_PLANNED = 3  # words: under this a line of the writer's isn't a sentence ('4.', '"..."', 'Luka writes:')
OFF_PLAN = 0.7  # of a plan's words, or of its sentences' length, under which a rewrite is asked for again
_CUT_OFF = re.compile(r"\b(?:the|a|an|of|and|their|which)\.(?=\s|$)")
ALL_OF_PLAN = ("That came back short of its sentence plan. Write it again with every line at its full length, each "
               "a whole sentence: join clauses with commas and 'and' to reach the length, and never end a sentence "
               "on a word like 'the' or leave a word on its own.")
MARK_NAMES = ((",", "comma"), ("(", "bracket pair"), (";", "semicolon"), (":", "colon"), ("?", "question mark"))


def structure(plan) -> str:
    """How closely the plan's rewrites follow their originals: "writer" without the writer's sentence
    shapes to borrow (no profile, or a visitor's own on a public deploy) works as "same"."""
    chosen = getattr(plan, "structure", "free")
    return "same" if chosen == "writer" and not getattr(plan, "shapes", None) else chosen


def follows(plan, job: dict) -> bool:
    """Whether a block's new text follows its original as the plan says. In a whole-document rewrite an
    answer box's existing answer is an original like any text block, or those few words read as the
    model's own (on one report 537 words of them took the whole from 18% to 28%); filled on their own,
    answer boxes get new answers instead."""
    if structure(plan) == "free":
        return False
    return job["kind"] != "answer_box" or bool(getattr(plan, "rewrite_all_blocks", False) and job["previous"].strip())


def sentence_shape(sentence: str) -> tuple[int, str]:
    """A sentence's length in words and its punctuation, the way a sentence plan lists them."""
    marks = [f"{count} {name}{'s' if count > 1 else ''}" for mark, name in MARK_NAMES
             if (count := sentence.count(mark))]
    return count_words(sentence), ", ".join(marks)


def paragraph_shapes(paragraphs: Iterable[str]) -> list[list[tuple[int, str]]]:
    """Each paragraph's sentences as shapes (sentence_shape), the skeletons "writer" borrows from."""
    return [[sentence_shape(text[s.start:s.end]) for s in split_sentences(text)] for text in paragraphs]


def _sentences_only(shapes: list[list[tuple[int, str]]]) -> list[list[tuple[int, str]]]:
    """Shapes with what the splitter took for a sentence left out: a list number, a quotation cut to
    '"..."', a label. Their one-word lines in a plan had models write 'Veterans.' and 'Power.'"""
    return [[shape for shape in paragraph if shape[0] >= MIN_PLANNED] for paragraph in shapes]


def borrowed_skeleton(shapes: list[list[tuple[int, str]]], words: int, key: str) -> list[tuple[int, str]] | None:
    """A run of consecutive sentences from one of the writer's paragraphs, `words` long in all to within a
    quarter, picked by the block's `key` so a block keeps its plan when it is written again. None when
    nothing the writer wrote fits."""
    fits = []
    for paragraph in _sentences_only(shapes):
        for start in range(len(paragraph)):
            total = 0
            for end in range(start, len(paragraph)):
                total += paragraph[end][0]
                if total > 1.25 * words:
                    break
                if total >= 0.75 * words:
                    fits.append(paragraph[start:end + 1])
    return fits[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(fits)] if fits else None


def _skeleton(plan, job: dict) -> list[tuple[int, str]]:
    """The sentence plan of a "writer" rewrite: borrowed from the writer, or the block's own when the
    writer has nothing that fits."""
    words = round(count_words(job["previous"]) * getattr(plan, "stretch", 1.0))
    return (borrowed_skeleton(getattr(plan, "shapes", None) or [], words, job.get("key", job["label"]))
            or paragraph_shapes([job["previous"]])[0])


def off_plan(plan, job: dict, text: str) -> bool:
    """Whether a "writer" rewrite came back short of its plan: too few words, sentences well under the
    plan's length, or one cut off ('...the fall of the.'). Models write about half of a plan's words."""
    if structure(plan) != "writer" or not follows(plan, job) or not text.strip():
        return False
    skeleton = _skeleton(plan, job)
    planned, words = sum(length for length, _ in skeleton), count_words(text)
    sentences = len(split_sentences(text)) or 1
    return bool(_CUT_OFF.search(text) or words < OFF_PLAN * planned
                or words / sentences < OFF_PLAN * planned / len(skeleton))


def again_note(plan, job: dict, text: str) -> str:
    """What to tell the model when a block is to be written again: nothing when it needn't be."""
    return AGAIN if copied(job["previous"], text) else ALL_OF_PLAN if off_plan(plan, job, text) else ""


def original_text(plan, job: dict) -> str:
    """A text block's original as the prompt shows it. When the rewrite follows it, its sentences are
    numbered with their lengths, a blank line between paragraphs, so the model can match each one. In
    "writer" it comes as it is, with the sentence plan borrowed from the writer (plan.shapes), or the
    original's own when the writer has nothing that fits."""
    if not follows(plan, job):
        return job["previous"]
    if structure(plan) == "writer":
        skeleton = _skeleton(plan, job)
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
    if not follows(plan, job) or not text.strip():
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
    a block short of its plan asked for again too (off_plan), and the blocks a reply left out, to be
    written on their own. With `plan.mask_facts`, the blocks go out with their names and numbers as
    placeholders (FactMask) and come back with them restored: a reply that loses one is asked for again
    with the others, and one that loses it twice is written on its own, as a left-out block is."""
    texts, alone, pending, notes = {}, [], group, ""
    originals = {job["key"]: job for job in group}
    mask = None
    if getattr(plan, "mask_facts", False):
        mask = FactMask(plan.context_text, *(job["previous"] for job in group))
        pending = [{**job, "previous": mask.hide(job["previous"]),
                    **({"anchor": mask.hide(job["anchor"])} if job.get("anchor") else {})} for job in group]
    for attempt in range(2):
        prompt = group_prompt(plan, pending) + notes
        if mask:
            prompt += mask.legend([job["previous"] for job in pending])
        parsed = parse_group_reply(ask(prompt, pending), [f"B{number}" for number in range(1, len(pending) + 1)])
        copies, short, lost = [], [], []
        for number, job in enumerate(pending, 1):
            if not (text := follow_original(plan, job, clean(parsed.get(f"B{number}", "")))):
                if job["key"] not in texts:
                    alone.append(originals[job["key"]])
                continue
            if mask and mask.lost(job["previous"], text):
                lost.append(job)
                continue
            if (attempt and job["key"] in texts and off_plan(plan, job, text)  # a lost block has no first text
                    and count_words(text) < count_words(texts[job["key"]])):
                continue  # asked again and came back shorter: the first was nearer the plan
            texts[job["key"]] = text
            if copied(job["previous"], text):
                copies.append(job)
            elif off_plan(plan, job, text):
                short.append(job)
        if not copies and not short and not lost:
            break
        if attempt:
            alone += [originals[job["key"]] for job in lost if job["key"] not in texts]
        pending = copies + short + lost
        notes = ((f"\n\n{AGAIN_ALL}" if copies else "") + (f"\n\n{ALL_OF_PLAN}" if short else "")
                 + (f"\n\n{LOST}" if lost else ""))
    return {key: mask.restore(text) for key, text in texts.items()} if mask else texts, alone


# Told to use plain words, a model doesn't: its vocabulary stayed about as varied whatever the prompt said
# (MATTR 0.80 against the writer's 0.68). Shown which of its own words the writer never uses, it can replace them.
# It does, but with wrong words ('printer' became 'bad machine'), so only eval/tune_prompts.py uses this.
_LOWER_WORD = re.compile(r"(?<![\w'\u2019-])[a-z]{4,}(?![\w'\u2019-])")
_ENDINGS = ("ing", "ed", "es", "ly", "s", "d")
MOST_LISTED = 40
PLAIN_WORDS = ("Below is a finished text and words in it that this writer never uses. Write the text again, changing "
               "only those words: use a plain word the writer would use, and where you can, one already in the "
               "text, repeating it rather than finding a new one. Keep the facts, the names, the sentences, the "
               "spelling and the punctuation as they are, and keep a term when the subject has no other word for "
               "it. Reply with the text only.\n\nWords to replace: {words}\n\nText:\n{text}")


def unfamiliar_words(text: str, vocabulary: set[str]) -> list[str]:
    """The text's lowercase words the writer has never written, in any common form, in order. Names are
    capitalized, so they stay out."""
    def known(word: str) -> bool:
        stems = {word[:-len(end)] for end in _ENDINGS if word.endswith(end)}
        return word in vocabulary or any(stem in vocabulary or stem + "e" in vocabulary for stem in stems)

    return list(dict.fromkeys(word for word in _LOWER_WORD.findall(text) if not known(word)))


def plainer_words(text: str, vocabulary: set[str], ask: Callable[[str], str],
                  clean: Callable[[str], str]) -> tuple[str, int, int]:
    """The text with the words the writer never uses replaced (one request), and how many there were before
    and after. The reply is kept only if it has fewer of them and about the same length and sentences,
    since a model asked to change words sometimes rewrites the whole text."""
    before = unfamiliar_words(text, vocabulary)
    if not vocabulary or not before:
        return text, len(before), len(before)
    new = clean(ask(PLAIN_WORDS.format(words=", ".join(before[:MOST_LISTED]), text=text)))
    after = unfamiliar_words(new, vocabulary)
    sentences = len(split_sentences(new)) / max(1, len(split_sentences(text)))
    if len(after) >= len(before) or not 0.85 <= count_words(new) / count_words(text) <= 1.15 \
            or not 0.85 <= sentences <= 1.15:
        return text, len(before), len(before)
    return new, len(before), len(after)


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
    shapes = _sentences_only(shapes)
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
# Revisions came back short too: on 6-7 Nov 634 of 4,378 tries were dropped as under 0.6 of their passage's length
# (median 0.49; a quarter of gemini-3.5-flash-lite's), and those kept ran a median 0.88. Short also means said less.
REVISE_STRETCH = 1.25


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


def restyle(draft: str, shapes: list, ask: Callable[[str, list[dict]], str], clean: Callable[[str], str],
            candidates: int = 1, score: Callable[[str], float | None] | None = None) -> str:
    """A draft rewritten passage by passage in the writer's sentences (the "writer" structure), as a
    document rewrite does, the passages joined back into the draft's paragraphs. Written from a prompt
    alone, a sample keeps the model's own plan of what to say, which a detector reads whatever voice
    it is told to write in; rewriting a draft in borrowed skeletons breaks that plan up the way the
    document rewrites that get past the detector do. A passage the reply leaves out stays as drafted. With
    `candidates` and `score`, one request asks for that many versions of each passage, in different plans, and the
    detector picks the lowest-scoring of each, so the first draft needs fewer revision rounds."""
    jobs = draft_jobs(draft)
    if not jobs:
        return draft
    plan = SimpleNamespace(prompt="", context_text=draft, blocks=[], answer_slots=[], match_existing_lengths=True,
                           answer_words=100, structure="writer", shapes=shapes, stretch=RESTYLE_STRETCH)
    tries = candidates if score and candidates > 1 else 1
    versions = _written_tries(plan, lambda attempt: [{**job, "key": f"{job['key']} try {attempt}"} for job in jobs],
                              ask, clean, tries)

    def assembled(chosen: dict[str, str]) -> str:
        paragraphs, passage = [], 0
        for part in (part.strip() for part in re.split(r"\n\s*\n", draft) if part.strip()):
            pieces = []
            while passage < len(jobs) and jobs[passage]["previous"] in part:
                pieces.append(chosen.get(jobs[passage]["key"]) or jobs[passage]["previous"])
                passage += 1
            paragraphs.append(" ".join(pieces) or part)
        return "\n\n".join(paragraphs)

    chosen = {key: texts[0] for key, texts in versions.items() if texts}
    best = score(assembled(chosen)) if tries > 1 else None
    if best is None:
        return assembled(chosen)
    for key, texts in versions.items():
        for text in texts[1:]:
            result = score(assembled({**chosen, key: text}))
            if result is not None and result < best:
                best, chosen = result, {**chosen, key: text}
    return assembled(chosen)



REVISE_PATIENCE = 2  # rounds in a row without a lower score before revising gives up: rounds cost a request each
WIN_BELOW = 0.10  # the bar: at this AI likelihood or above the detector has won, under it the generator has
REVISED_BLOCKS = 8  # a document's blocks revised per round, each costing one detector pass on the whole text
MIN_REVISED_WORDS = 15  # a shorter block barely moves a whole document's score
# Told what the detector flagged, in the same group request as any other rewrite.
REVISE = ("A detector reads the passages below as machine-written. Write each again the way the writer would "
          "really put it, not as a tidy summary: keep its point, but trade a general statement for one concrete, "
          "particular detail, use the writer's plain, specific words over safe, common ones, and leave out "
          "signposting, wrap-up lines and balanced pairs or lists.")
# As REVISE, for a text that has to keep saying what it said: left to it, revision drops names and numbers.
REVISE_KEEPING = (REVISE + " Keep every name, date and number the passage gives, and add none it doesn't. Leave out "
                  "the habits that give AI writing away (from github.com/blader/humanizer): 'not X but Y', a short "
                  "closing line that repeats the point, things grouped in threes when the meaning has no three "
                  "parts, inflated importance, a trailing '-ing' clause, and comments on the writing itself such as "
                  "'it is evident that' or 'this shows'. Where the style profile says otherwise, it wins.")
REVISE_CANDIDATES = 3  # the rewrites a passage gets each round, the best kept: one request each per group of passages
# Stock phrases, only to choose between candidates a detector scores alike (from ClearRewriter).
_STOCK = ("furthermore", "moreover", "additionally", "in conclusion", "it is important to note", "delve", "tapestry",
          "navigate the", "underscore", "multifaceted", "unprecedented", "pivotal role", "foster", "leverag",
          "in the realm of", "a testament to", "crucial", "comprehensive", "facilitate", "utilize", "it is evident")
_NEGATIONS = {"not", "no", "never", "none", "nor", "cannot", "without", "neither", "nobody", "nothing"}
_PERSONAL = {"i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours"}
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
STYLE_SLACK = 0.05  # how much further from the profile (StyleYardstick.text_distance) a revised text may drift


FACTS_KEPT = 0.8  # the share of a passage's names and numbers a revision has to keep
_FACT = re.compile(r"(?<![.!?:\"\u201c] )(?<!^)\b[A-Z][a-z]{2,}\b|\b\d[\d,.]*\d\b")


def keeps_facts(passage: str, revised: str, share: float = FACTS_KEPT) -> bool:
    """Whether a revision still says what the passage says: most of its numbers of two digits or more and its
    capitalized words other than those opening a sentence. A prompt alone left revision dropping them."""
    terms = {term.rstrip(".,") for term in _FACT.findall(passage.strip())}
    return not terms or sum(term in revised for term in terms) >= share * len(terms)


def drifted(passage: str, revised: str, original: str) -> bool:
    """Whether a revision drops a name or number the writer's original has and the passage still says. keeps_facts
    lets each revision lose a fifth of a passage's facts, and round after round that compounded: revisions that took
    8 or 9 rounds kept 12-19% of the original's names and numbers (the 6-7 Nov overnight run). Holding what the
    original and the passage share keeps any number of rounds from wearing it away."""
    held = {term.rstrip(".,") for term in _FACT.findall(passage.strip())} & {
        term.rstrip(".,") for term in _FACT.findall(original.strip())}
    return any(term not in revised for term in held)


def stock_phrases(text: str) -> int:
    low = text.lower()
    return sum(phrase in low for phrase in _STOCK)


def meaning_problem(passage: str, revised: str, context: str = "") -> str | None:
    """What keeps a rewrite from counting as the same passage, after ClearRewriter's check: `facts` (keeps_facts),
    `length` (about the same), `new_fact` (a name or number that isn't in the passage or in `context`, the document
    around it), `negation` (more than one gained or lost) or `person` ('I', 'we' or 'you' the passage doesn't have)."""
    if not keeps_facts(passage, revised):
        return "facts"
    if not 0.6 <= len(revised) / max(1, len(passage)) <= 1.6:
        return "length"
    known = passage + " " + context
    if any(term.rstrip(".,") not in known for term in _FACT.findall(revised.strip())):
        return "new_fact"

    def words(text: str) -> list[str]:
        return [word.split("'")[0] if word.endswith(("'re", "'ll", "'ve", "'m", "'d")) else word
                for word in _WORD.findall(text.lower().replace("\u2019", "'"))]

    before, after = words(passage), words(revised)
    negations = [sum(word in _NEGATIONS or word.endswith("n't") for word in side) for side in (before, after)]
    if abs(negations[0] - negations[1]) > 1:
        return "negation"
    return "person" if (set(after) - set(before)) & _PERSONAL else None


def _persons(text: str) -> set[str]:
    """The 'I', 'we' and 'you' words a text uses."""
    return set(_WORD.findall(text.lower().replace("\u2019", "'"))) & _PERSONAL


def meaning_kept(passage: str, revised: str, context: str = "") -> bool:
    return meaning_problem(passage, revised, context) is None


def _written_tries(plan, jobs_for: Callable[[int], list[dict]], ask, clean, tries: int) -> dict[str, list[str]]:
    """What `tries` rewrites of the same passages came back as, by the part of each job's key before its ' try '. All
    in one request, each in its own plan: asking once per try sent the whole profile and document again each time."""
    written: dict[str, list[str]] = {}
    jobs = [job for attempt in range(tries) for job in jobs_for(attempt)]
    results = rewrite_group(plan, jobs, ask, clean)[0]
    for job in jobs:
        if job["key"] in results:
            written.setdefault(job["key"].split(" try ")[0], []).append(results[job["key"]])
    return written


def style_guard(distance: Callable[[str], float | None], text: str,
                slack: float = STYLE_SLACK) -> Callable[[str], bool]:
    """Whether a revision of `text` stays as close to the style profile as `text` is, within `slack`: revising
    only for the detector's score moves a text away from the writer. Anything goes when it can't be measured."""
    start = distance(text)

    def fits(candidate: str) -> bool:
        reached = distance(candidate)
        return start is None or reached is None or reached <= start + slack
    return fits


def _stalled(history: list[float]) -> bool:
    """Whether the last `REVISE_PATIENCE` rounds of revising lowered nothing."""
    return len(history) > REVISE_PATIENCE and history[-1] >= history[-1 - REVISE_PATIENCE]


def passage_scores(spans: list[tuple[int, int]], sentences: list) -> list[float]:
    """Each passage's mean sentence AI likelihood from a detector Report's `sentences` (0 with none)."""
    scores = []
    for start, end in spans:
        inside = [s.ai_probability for s in sentences if start <= s.start < end]
        scores.append(sum(inside) / len(inside) if inside else 0.0)
    return scores


def revise_flagged(text: str, analyze: Callable[[str], object], ask: Callable[[str, list[dict]], str],
                   clean: Callable[[str], str], shapes: list, rounds: int = 2, target: float = WIN_BELOW,
                   fits: Callable[[str], bool] | None = None, prompt: str = REVISE,
                   facts: bool = False, candidates: int = 1, above: float | None = None,
                   max_requests: int | None = None, mask: bool = False) -> tuple[str, list[float | None]]:
    """A text revised where the detector reads it as AI: each round rewrites its most AI-like passages
    (by sentence score, at most half of them) in new sentence plans borrowed from the writer, told what
    was flagged (REVISE), and keeps each rewritten passage only if the whole text then scores lower,
    trying them most flagged first. Kept or dropped as a whole, a round lost every good passage to one
    bad one, and the samples that stayed flagged were the ones whose rounds were dropped. Stops once it
    scores under `target`. Rewriting a whole draft again keeps the model's own plan of what to say;
    rewriting only where the detector reads it breaks that plan up where it shows. `analyze` is the
    detector's (with sentence scores), run once per rewritten passage. With `fits` (style_guard) a passage
    that takes the text away from the style profile is dropped too. Gives the kept text, normalized as the
    detector read it, and its AI likelihood after each round, the first before any. With `candidates`, a
    passage gets that many rewrites a round, in different plans, and the lowest-scoring that passes is the one tried.
    With `above`, every passage scoring at least that is
    rewritten, no others, and revising ends when none is left, whatever the whole text scores. `max_requests` stops
    it before a round once that many requests have gone. With `mask`, names and numbers go out as placeholders
    (FactMask)."""
    sent = 0
    inner_ask = ask

    def ask(prompt: str, jobs: list[dict]) -> str:
        nonlocal sent
        sent += 1
        return inner_ask(prompt, jobs)

    report = analyze(text)
    if report.ai_probability is None:
        return text, [None]
    best, history = report, [report.ai_probability]
    for round_ in range(1, rounds + 1):
        if (above is None and best.ai_probability < target) or (max_requests is not None and sent >= max_requests):
            break
        base, spans = best.text, passage_spans(best.text)
        if not spans:
            break
        scores = passage_scores(spans, best.sentences)
        order = sorted(range(len(spans)), key=lambda index: -scores[index])
        if above is None:
            chosen = [index for index in order if scores[index] >= target][:max(1, len(spans) // 2)] or order[:1]
        else:
            chosen = [index for index in order if scores[index] >= above]
            if not chosen:
                break
        def jobs(attempt: int, chosen=chosen, base=base, spans=spans, round_=round_) -> list[dict]:
            return [{"kind": "text_block", "label": f"Passage {index + 1}", "question": "",
                     "previous": base[slice(*spans[index])],
                     "key": f"{index + 1:03d} revise {round_} try {attempt}"} for index in sorted(chosen)]

        plan = SimpleNamespace(prompt=prompt, context_text=base, blocks=[], answer_slots=[],
                               match_existing_lengths=True, answer_words=100, structure="writer" if shapes else "free",
                               shapes=shapes, mask_facts=mask, stretch=REVISE_STRETCH)  # models write short
        written = _written_tries(plan, jobs, ask, clean, candidates)
        passages = [base[start:end] for start, end in spans]
        gaps = [base[:spans[0][0]]] + [base[end:after[0]] for (_, end), after in pairwise(spans)] \
            + [base[spans[-1][1]:]]
        for index in chosen:
            kept = None
            for new in written.get(f"{index + 1:03d} revise {round_}", []):
                new = new.strip()
                if not new or new == passages[index]:
                    continue
                if facts and not meaning_kept(passages[index], new, base):
                    continue
                trial = [*passages[:index], new, *passages[index + 1:]]
                joined = "".join(gap + passage for gap, passage in zip(gaps, [*trial, ""], strict=True))
                if fits and not fits(joined):
                    continue
                candidate = analyze(joined)
                if candidate.ai_probability is None:
                    continue
                rank = (round(candidate.ai_probability, 2), stock_phrases(new))
                if kept is None or rank < kept[0]:
                    kept = (rank, candidate, trial)
            if kept and kept[1].ai_probability < best.ai_probability:
                best, passages = kept[1], kept[2]
        history.append(best.ai_probability)
        if _stalled(history):
            break
    return best.text, history


def document_windows(detector) -> Callable[[str], list[tuple[int, int, float]]]:
    """A detector's document windows of a text, the ones its score is computed from: each one's character
    span and its classifiers' mean log-odds. Only reads the detector, through what `features` uses."""
    def windows(text: str) -> list[tuple[int, int, float]]:
        clean, _, sentences = detector._prepare(text)
        cfg = detector.config
        spans = sentence_windows(sentences, cfg.doc_window_words, cfg.doc_stride_words)
        scores, _ = detector._engine_scores(clean, spans)
        names = [engine.name for engine in detector.classifiers]
        return [(span.start, span.end, sum(float(scores[name][number]) for name in names) / len(names))
                for number, span in enumerate(spans)]
    return windows


def window_heat(spans: dict[str, tuple[int, int]], windows: list[tuple[int, int, float]]) -> dict[str, float]:
    """Each block's reading from the document windows: the highest score among the windows holding most of
    it, or among those touching it when none does."""
    heat = {}
    for key, (start, end) in spans.items():
        touching = [(min(end, stop) - max(start, begin), score) for begin, stop, score in windows
                    if begin < end and start < stop]
        holding = [score for overlap, score in touching if 2 * overlap >= end - start]
        heat[key] = max(holding or [score for _, score in touching], default=float("-inf"))
    return heat


def _passages(texts: dict[str, str]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """A document's blocks cut into passages, by "block#number" in document order: each passage's text (normalized,
    as the detector reads it), what follows it, and the block it belongs to."""
    shown, gaps, blocks = {}, {}, {}
    for key, text in texts.items():
        plain = normalize(text)[0]
        spans = passage_spans(plain) or [(0, len(plain))]
        for number, (start, end) in enumerate(spans):
            name = f"{key}#{number}"
            shown[name], blocks[name] = plain[start:end], key
            gaps[name] = plain[end:spans[number + 1][0]] if number + 1 < len(spans) else "\n\n"
    return shown, gaps, blocks


def revise_blocks(texts: dict[str, str], analyze: Callable[..., object], ask: Callable[[str, list[dict]], str],
                  clean: Callable[[str], str], shapes: list, rounds: int = 2, target: float = WIN_BELOW,
                  windows: Callable[[str], list[tuple[int, int, float]]] | None = None,
                  fits: Callable[[str], bool] | None = None, prompt: str = REVISE, facts: bool = False,
                  candidates: int = 1, progress: Callable[[dict], None] | None = None,
                  stats: Counter | None = None, mask: bool = False, trace=None,
                  score_many: Callable[[list[str]], list[float | None]] | None = None,
                  saved: Callable[[dict[str, str], list[float | None]], None] | None = None,
                  ) -> tuple[dict[str, str], list[float | None]]:
    """A document's rewritten blocks revised where the detector reads the whole as AI, as revise_flagged does
    a sample, keeping each block in its place in the document. `texts` holds each block's text by key, in
    document order. A long block is cut into passages (passage_spans), since whole it gets one all-or-nothing
    try a round and documents of long paragraphs stalled. Each round rewrites the most AI-like passages of
    `MIN_REVISED_WORDS` or more (at most `REVISED_BLOCKS`), and keeps each only if the whole document then
    scores lower. Stops once it scores under `target`. With `windows` (document_windows) the blocks of the
    hottest document window go first,
    sentence scores only breaking ties, since the worst such window is what the document's score reads.
    With `fits` (style_guard) a block that takes the whole away from the style profile is dropped too, and with
    `facts` (meaning_kept) one that loses the passage's names and numbers or changes its meaning. With `candidates`,
    a passage gets that many rewrites a round, in different plans, and the lowest-scoring that passes is the one tried.
    `progress`, if given, is told the round, rounds, passages tested, passages chosen and current AI likelihood
    as they change, since one round can take minutes.
    `stats`, if given, counts what became of each rewrite: `unchanged`, `facts` and `style` (dropped by those checks),
    `scored` (scored against the whole), then per passage `kept` or `not_lower`.
    With `mask`, names and numbers go out as placeholders (FactMask).
    With `score_many` (CachedScorer.many), a passage's rewrites are scored against the whole in one batch, the model
    run only on document windows it hasn't seen: one passage changes two or three of a long document's windows, and
    scoring every window of every try made a round of a 2,400-word document take several minutes on a CPU. The
    starting figure comes from it too, so before and after are measured alike. Left out with `trace`, which records
    each try's windows.
    `saved`, if given, is told the changed blocks and the history after every round, so a quit or a crash keeps the
    rounds done.
    Gives the changed blocks' texts by key, and the AI likelihood after each round, the first before any."""
    tell = progress or (lambda _: None)
    stats = Counter() if stats is None else stats
    shown, gaps, blocks = _passages(texts)

    def whole(parts: dict[str, str]) -> str:
        return "".join(parts[name] + gaps[name] for name in parts).removesuffix("\n\n")

    def changed_blocks() -> dict[str, str]:
        return {key: "".join(shown[name] + gaps[name] for name in shown if blocks[name] == key).removesuffix("\n\n")
                for key in texts if key in touched}

    cached = score_many if trace is None else None
    report = analyze(whole(shown))
    if report.ai_probability is None:
        return {}, [None]
    best = report.ai_probability if cached is None else cached([whole(shown)])[0]
    history, touched = [best], set()
    if trace:
        trace.version("revise_start", whole(shown), report)
    for round_ in range(1, rounds + 1):
        if best < target:
            break
        spans, start = {}, 0
        for name, text in shown.items():
            spans[name] = (start, start + len(text))
            start += len(text) + len(gaps[name])
        scores = dict(zip(spans, passage_scores(list(spans.values()), report.sentences), strict=True))
        long_enough = [name for name in shown if count_words(shown[name]) >= MIN_REVISED_WORDS]
        if windows:
            heat = window_heat(spans, windows(whole(shown)))
            chosen = sorted(long_enough, key=lambda name: (-heat[name], -scores[name]))[:REVISED_BLOCKS]
        else:
            order = sorted(long_enough, key=lambda name: -scores[name])
            chosen = [name for name in order if scores[name] >= target][:REVISED_BLOCKS] or order[:1]
        if not chosen:
            break
        tell({"round": round_, "rounds": rounds, "tested": 0, "passages": len(chosen), "likelihood": best})

        def jobs(attempt: int, chosen=chosen, shown=shown, round_=round_) -> list[dict]:
            return [{"kind": "text_block", "label": f"Passage {number}", "question": "", "previous": shown[name],
                     "key": f"{name} revise {round_} try {attempt}"} for number, name in enumerate(chosen, 1)]

        plan = SimpleNamespace(prompt=prompt, context_text=whole(shown), blocks=[], answer_slots=[],
                               match_existing_lengths=True, answer_words=100, structure="writer" if shapes else "free",
                               shapes=shapes, mask_facts=mask)
        written = _written_tries(plan, jobs, ask, clean, candidates)
        for tested, name in enumerate(chosen, 1):
            kept, passing = None, []
            for text in written.get(f"{name} revise {round_}", []):
                new = normalize(text)[0]
                if not new or new == shown[name]:
                    stats["unchanged"] += 1
                    continue
                problem = meaning_problem(shown[name], new, whole(shown)) if facts else None
                if problem:
                    stats["facts"] += 1
                    stats[f"facts:{problem}"] += 1
                    if trace:
                        trace.event("candidate", round=round_, passage=name, before=shown[name], text=new,
                                    status="dropped", reason=problem)
                    continue
                trial = {**shown, name: new}
                if fits and not fits(whole(trial)):
                    stats["style"] += 1
                    if trace:
                        trace.event("candidate", round=round_, passage=name, before=shown[name], text=new,
                                    status="dropped", reason="style")
                    continue
                passing.append(new)
            stats["scored"] += len(passing)
            if cached is not None:
                found = cached([whole({**shown, name: new}) for new in passing]) if passing else []
            else:
                found = []
                for new in passing:
                    scored = analyze(whole({**shown, name: new}), sentences=False)
                    found.append(scored.ai_probability)
                    if trace:
                        trace.event("candidate", round=round_, passage=name, before=shown[name], text=new,
                                    status="scored", score=scored.ai_probability, passage_score=scores[name],
                                    best_before=best, windows=scored.windows)
            for new, candidate in zip(passing, found, strict=True):
                if candidate is None:
                    continue
                rank = (round(candidate, 2), stock_phrases(new))
                if kept is None or rank < kept[0]:
                    kept = (rank, candidate, new)
            if trace:
                trace.event("passage", round=round_, passage=name, kept=bool(kept and kept[1] < best),
                            lowest=kept[1] if kept else None, best_before=best)
            if kept and kept[1] < best:
                best, shown[name] = kept[1], kept[2]
                touched.add(blocks[name])
                stats["kept"] += 1
            elif kept:
                stats["not_lower"] += 1
            tell({"round": round_, "rounds": rounds, "tested": tested, "passages": len(chosen), "likelihood": best})
        history.append(best)
        if saved:
            saved(changed_blocks(), list(history))
        if trace:
            trace.version("revise_round", whole(shown), round=round_, score=best,
                          chosen=[[name, round(scores[name], 4)] for name in chosen])
        if _stalled(history):
            break
        if best >= target and round_ < rounds:
            report = analyze(whole(shown))  # sentence scores for the next round's choice
    return changed_blocks(), history


CONTEXT_WORDS = 350  # a passage is scored inside about this many words around it: a document window's size
# Told to one half of a passage's tries: the writer's original reads as theirs, and the draft drifted from it.
ANCHORED = ("Start from the writer's original of this part, given with it: keep its sentences, their order and most "
            "of its words, fit it to what this passage covers, and reword about one phrase in four, so it isn't "
            "copied word for word.")
REVISE_GROUP = 6  # passages in one revise request, each with all its tries
ANCHORED_CANDIDATES = 4  # tries a passage gets a round in revise_anchored, half of them from the original
# ...and in its first round, to win in fewer rounds: every try of a group comes back in one request, so more cost
# tokens and detector time, not requests. On 6-7 Nov a try lowered its document 25% of the time: if tries were
# independent, 4 would fix about 68% of flagged passages and 8 about 90%.
ANCHORED_FIRST_ROUND = 8


def chunk_scorer(detector) -> Callable[[list[str]], list[float | None]]:
    """The AI likelihood of each chunk of text read as one document window, as the sentence highlights read
    theirs, remembered by text so the model runs once per chunk. A build with Binoculars scores each through
    `detector.score`, which reads the chunk's own windows."""
    import numpy as np

    seen: dict[str, float | None] = {}

    def score(chunks: list[str]) -> list[float | None]:
        missing = list(dict.fromkeys(chunk for chunk in chunks if chunk not in seen))
        if missing and getattr(detector, "binoculars", None) is None:
            logits = {engine.name: np.asarray(engine.window_logits(missing)) for engine in detector.classifiers}
            rows = [detector._window_features(logits, number) for number in range(len(missing))]
            seen.update(zip(missing, (float(p) for p in detector._probabilities(rows)), strict=True))
        elif missing:
            seen.update((chunk, detector.score(chunk)) for chunk in missing)
        return [seen[chunk] for chunk in chunks]
    return score


def revise_anchored(texts: dict[str, str], originals: dict[str, str], score_chunks: Callable[[list[str]], list],
                    score: Callable[[str], float | None], ask: Callable[[str, list[dict]], str],
                    clean: Callable[[str], str], shapes: list | None = None, rounds: int = 3,
                    target: float = WIN_BELOW, fits: Callable[[str], bool] | None = None,
                    prompt: str = REVISE_KEEPING, facts: bool = True, candidates: int = ANCHORED_CANDIDATES,
                    progress: Callable[[dict], None] | None = None, stats: Counter | None = None,
                    mask: bool = False, saved: Callable[[dict[str, str], list[float | None]], None] | None = None,
                    parallel: int = 1, trace=None, first_round: int = ANCHORED_FIRST_ROUND
                    ) -> tuple[dict[str, str], list[float | None]]:
    """A document's rewritten blocks revised back toward the writer's originals where the detector reads them as AI.

    revise_blocks capped a round at a few passages, chosen by sentence scores, and wrote every try in a borrowed
    sentence plan, so a long document took many rounds and drifted from its original. Here each round takes every
    passage that reads as AI (at `target` / 2 or more) inside the `CONTEXT_WORDS` around it, as the document's
    windows read it, since the score is their mean and maximum. Each such passage gets `candidates` tries
    (`first_round` in the first, to win in fewer rounds; plans stretched by REVISE_STRETCH): half start
    from the writer's original of its block (`originals`, by key; ANCHORED), the rest rewrite it as revise_blocks
    does, in the writer's sentence plans when `shapes` are given. A try is scored on its own surroundings
    (`score_chunks`, chunk_scorer), and the lowest is kept if it beats the passage there; among tries that already
    read as human (under `target` / 2) the one keeping most of the original's words wins. A try that leaves its block
    a copy of the original (COPIED) is dropped, as are those failing `facts` (meaning_problem, and `drifted` against the
    original) or `fits` (style_guard).
    The whole document is scored with `score` after each round, and a round that raised it is undone. Requests go
    `REVISE_GROUP` passages at a time, `parallel` at once. `progress`, `stats` and `saved` are as in revise_blocks,
    `stats` adding `copied`, and `trace` (eval/trace.py) records each round, passage and try with why it was kept
    or dropped. Gives the changed blocks' texts by key, and the AI likelihood after each round, the first
    before any."""
    tell = progress or (lambda _: None)
    stats = Counter() if stats is None else stats
    shown, gaps, blocks = _passages(texts)
    names, touched = list(shown), set()
    good_enough = target / 2

    def whole(parts: dict[str, str]) -> str:
        return "".join(parts[name] + gaps[name] for name in names).removesuffix("\n\n")

    def block_text(parts: dict[str, str], key: str) -> str:
        return "".join(parts[name] + gaps[name] for name in names if blocks[name] == key).removesuffix("\n\n")

    def changed_blocks() -> dict[str, str]:
        return {key: block_text(shown, key) for key in texts if key in touched}

    def around(parts: dict[str, str], name: str) -> tuple[int, int]:
        """The passages read with `name`: its neighbours on both sides until about CONTEXT_WORDS."""
        low = high = names.index(name)
        words = count_words(parts[name])
        while words < CONTEXT_WORDS and (low > 0 or high < len(names) - 1):
            if high < len(names) - 1:
                high += 1
                words += count_words(parts[names[high]])
            if words < CONTEXT_WORDS and low > 0:
                low -= 1
                words += count_words(parts[names[low]])
        return low, high

    def chunk(parts: dict[str, str], span: tuple[int, int]) -> str:
        return "".join(parts[name] + gaps[name] for name in names[span[0]:span[1] + 1]).strip()

    def closeness(parts: dict[str, str], key: str) -> float:
        original = originals.get(key, "")
        return rewrite_measures(original, block_text(parts, key)).get("words_kept", 0.0) if original.strip() else 0.0

    best = score(whole(shown))
    if best is None:
        return {}, [None]
    history = [best]
    if trace:
        trace.version("revise_start", whole(shown), score=best, loop="anchored")

    def note(name: str, round_: int, new: str, **fields) -> None:
        if trace:
            trace.event("candidate", round=round_, passage=name, before=shown[name], text=new, **fields)
    for round_ in range(1, rounds + 1):
        if best < target:
            break
        long_enough = [name for name in names if count_words(shown[name]) >= MIN_REVISED_WORDS]
        spans = {name: around(shown, name) for name in long_enough}
        local = dict(zip(long_enough, score_chunks([chunk(shown, spans[name]) for name in long_enough]),
                         strict=True))
        ranked = sorted((name for name in long_enough if local[name] is not None), key=lambda name: -local[name])
        hot = [name for name in ranked if local[name] >= good_enough] or ranked[:max(1, len(ranked) // 4)]
        if not hot:
            break
        tell({"round": round_, "rounds": rounds, "tested": 0, "passages": len(hot), "likelihood": best})
        context = whole(shown)
        failures: list[Exception] = []
        count = max(first_round, candidates) if round_ == 1 else candidates
        anchored_tries = (count + 1) // 2

        def request(group: list[str], anchored: bool, round_=round_, context=context, parts=shown,
                    failures=failures, count=count, anchored_tries=anchored_tries) -> dict[str, list[str]]:
            def jobs(attempt: int) -> list[dict]:
                number = attempt if anchored else anchored_tries + attempt
                return [{"kind": "text_block", "label": f"Passage {index}", "question": "", "previous": parts[name],
                         "key": f"{name} revise {round_} try {number}",
                         **({"anchor": originals[blocks[name]], "note": ANCHORED}
                            if anchored and originals.get(blocks[name], "").strip() else {})}
                        for index, name in enumerate(group, 1)]
            writer = bool(shapes) and not anchored
            plan = SimpleNamespace(prompt=prompt, context_text=context, blocks=[], answer_slots=[],
                                   match_existing_lengths=True, answer_words=100,
                                   structure="writer" if writer else "free", shapes=shapes or [], mask_facts=mask,
                                   stretch=REVISE_STRETCH)
            tries = anchored_tries if anchored else count - anchored_tries
            try:
                return _written_tries(plan, jobs, ask, clean, tries) if tries else {}
            except StopRequested:
                raise
            except Exception as exc:  # an overloaded model: the round goes on with the other requests' tries
                failures.append(exc)
                return {}

        groups = [hot[at:at + REVISE_GROUP] for at in range(0, len(hot), REVISE_GROUP)]
        written: dict[str, list[str]] = {}
        sides = [(group, side) for group in groups for side in (True, False)]
        with ThreadPoolExecutor(max(1, parallel)) as pool:
            for found in pool.map(lambda job: request(*job), sides):
                for key, versions in found.items():
                    written.setdefault(key, []).extend(versions)
        stats["requests_failed"] += len(failures)
        if failures and len(failures) == len(sides):
            raise failures[-1]  # nothing came back: no round to score
        before = dict(shown)
        for tested, name in enumerate(hot, 1):
            span = around(shown, name)
            current = score_chunks([chunk(shown, span)])[0]
            key, passing = blocks[name], []
            for text in written.get(f"{name} revise {round_}", []):
                new = normalize(text)[0]
                if not new or new == shown[name]:
                    stats["unchanged"] += 1
                    note(name, round_, new, status="dropped", reason="unchanged")
                    continue
                original = originals.get(key, "")
                problem = meaning_problem(shown[name], new, whole(shown) + "\n\n" + original) if facts else None
                if problem == "person" and _persons(new) <= _persons(shown[name] + " " + original):
                    problem = None  # the writer's original says it so: the draft had dropped it
                if not problem and facts and original.strip() and drifted(shown[name], new, original):
                    problem = "drift"
                if problem:
                    stats["facts"] += 1
                    stats[f"facts:{problem}"] += 1
                    note(name, round_, new, status="dropped", reason=problem)
                    continue
                trial = {**shown, name: new}
                if originals.get(key, "").strip() and copied(originals[key], block_text(trial, key)):
                    stats["copied"] += 1
                    note(name, round_, new, status="dropped", reason="copied")
                    continue
                if fits and not fits(whole(trial)):
                    stats["style"] += 1
                    note(name, round_, new, status="dropped", reason="style")
                    continue
                passing.append(trial)
            stats["scored"] += len(passing)
            found = score_chunks([chunk(trial, span) for trial in passing]) if passing else []
            kept = None
            for trial, reading in zip(passing, found, strict=True):
                note(name, round_, trial[name], status="scored", score=reading, passage_score=current,
                     closeness=closeness(trial, key))
                if reading is None:
                    continue
                rank = (round(max(reading, good_enough), 2), -closeness(trial, key), stock_phrases(trial[name]))
                if kept is None or rank < kept[0]:
                    kept = (rank, reading, trial[name])
            if trace:
                trace.event("passage", round=round_, passage=name, local_before=current,
                            kept=bool(kept and current is not None and kept[1] < current),
                            lowest=kept[1] if kept else None, best_before=best)
            if kept and current is not None and kept[1] < current:
                shown[name] = kept[2]
                stats["kept"] += 1
            elif kept:
                stats["not_lower"] += 1
            tell({"round": round_, "rounds": rounds, "tested": tested, "passages": len(hot), "likelihood": best})
        now = score(whole(shown))
        if now is None or now >= best:  # the document's own windows read the round as no better: undo it
            shown = before
            stats["rounds_undone"] += 1
        else:
            best = now
            touched.update(blocks[name] for name in names if shown[name] != before[name])
        history.append(best)
        if saved:
            saved(changed_blocks(), list(history))
        if trace:
            trace.version("revise_round", whole(shown), round=round_, score=best, undone=shown is before,
                          passages=[[name, round(local[name], 4)] for name in hot], failed=len(failures))
        if _stalled(history):
            break
    return changed_blocks(), history


_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_TOKEN = re.compile(r"\S+")


def selected_spans(text: str, sentences: list, chosen: Iterable[int], words: int = 80) -> list[tuple[int, int]]:
    """The passages to revise for the sentences a reader picked (indexes into a detector Report's `sentences`, whose
    offsets are into `text`): each run of neighbouring picked sentences in one paragraph, a run of more than about
    `words` words cut at a sentence end, so each passage fits a sentence plan borrowed from the writer."""
    spans: list[tuple[int, int]] = []
    count = 0
    previous = None
    for index in sorted(set(chosen)):
        if not 0 <= index < len(sentences):
            continue
        sentence = sentences[index]
        joined = (spans and previous == index - 1 and count < words
                  and not _BLANK_LINE.search(text[spans[-1][1]:sentence.start]))
        if joined:
            spans[-1] = (spans[-1][0], sentence.end)
            count += count_words(text[sentence.start:sentence.end])
        else:
            spans.append((sentence.start, sentence.end))
            count = count_words(text[sentence.start:sentence.end])
        previous = index
    return spans


def _surroundings(before: str, passage: str, after: str, words: int = CONTEXT_WORDS) -> str:
    """A passage with about `words` words of the text around it, half from each side where there is enough: what
    the detector's window would read it in."""
    need = max(0, words - count_words(passage))
    starts = [match.start() for match in _TOKEN.finditer(before)]
    ends = [match.end() for match in _TOKEN.finditer(after)]
    left = min(len(starts), max(need // 2, need - len(ends)))
    right = min(len(ends), need - left)
    return ((before[starts[-left]:] if left else "") + passage + (after[:ends[right - 1]] if right else "")).strip()


def revise_selected(text: str, spans: list[tuple[int, int]], score_chunks: Callable[[list[str]], list],
                    score: Callable[[str], float | None], ask: Callable[[str, list[dict]], str],
                    clean: Callable[[str], str], shapes: list | None = None, rounds: int = 3,
                    target: float = WIN_BELOW, fits: Callable[[str], bool] | None = None,
                    prompt: str = REVISE_KEEPING, facts: bool = True, candidates: int = REVISE_CANDIDATES,
                    mask: bool = False, parallel: int = 1, progress: Callable[[dict], None] | None = None,
                    stats: Counter | None = None) -> tuple[str, list[float | None], list[float | None]]:
    """`text` revised in the passages a reader chose (`spans`, character spans of `text`, as selected_spans gives),
    and nowhere else: the Detect page's Revise. As revise_anchored does a document's flagged passages, each round
    takes every chosen passage still reading as AI (at `target` / 2 or more) inside the `CONTEXT_WORDS` around it,
    asks for `candidates` tries of each, in the writer's sentence plans when `shapes` are given, drops a try that
    changes the passage's meaning (`facts`, meaning_problem) or leaves the style profile (`fits`), and keeps the one
    reading lowest in its surroundings if it beats the passage there. The whole text is scored with `score` after a
    round, and a round that raised it is undone. Stops when every chosen passage reads as human, after `rounds`, or
    after a round that kept nothing. `progress` hears each step: its "stage" is "asking" (waiting on the model),
    "scoring" (the detector reading the tries, `tested` of `passages` done) or "whole" (scoring the whole text). Gives
    the revised text, the whole text's AI likelihood after each round (the first before any), and each chosen
    passage's local AI likelihood at the end, in order."""
    tell = progress or (lambda _: None)
    stats = Counter() if stats is None else stats
    spans = sorted((start, end) for start, end in spans if text[start:end].strip())
    if not spans:
        return text, [score(text)], []
    gaps = [text[:spans[0][0]]] + [text[end:after[0]] for (_, end), after in pairwise(spans)] + [text[spans[-1][1]:]]
    parts = [text[start:end] for start, end in spans]
    good_enough = target / 2

    def whole(passages: list[str]) -> str:
        return "".join(gap + passage for gap, passage in zip(gaps, [*passages, ""], strict=True))

    def chunk(passages: list[str], index: int) -> str:
        before = "".join(gap + passage for gap, passage in zip(gaps[:index], passages[:index], strict=True))
        after = "".join(gap + passage for gap, passage in zip(gaps[index + 1:], [*passages[index + 1:], ""],
                                                              strict=True))
        return _surroundings(before + gaps[index], passages[index], after)

    best = score(text)
    if best is None:
        return text, [None], []
    history = [best]
    local = score_chunks([chunk(parts, index) for index in range(len(parts))])
    for round_ in range(1, rounds + 1):
        hot = [index for index in sorted(range(len(parts)), key=lambda i: -(local[i] or 0))
               if local[index] is not None and local[index] >= good_enough]
        if not hot:
            break
        tell({"round": round_, "rounds": rounds, "tested": 0, "passages": len(hot), "likelihood": best,
              "stage": "asking", "tries": candidates})
        context, asked = whole(parts), tuple(parts)
        failures: list[Exception] = []

        def request(group: list[int], round_=round_, context=context, failures=failures,
                    current=asked) -> dict[str, list[str]]:
            def jobs(attempt: int) -> list[dict]:
                return [{"kind": "text_block", "label": f"Passage {number}", "question": "", "previous": current[index],
                         "key": f"{index:03d} revise {round_} try {attempt}"} for number, index in enumerate(group, 1)]
            plan = SimpleNamespace(prompt=prompt, context_text=context, blocks=[], answer_slots=[],
                                   match_existing_lengths=True, answer_words=100,
                                   structure="writer" if shapes else "free", shapes=shapes or [], mask_facts=mask)
            try:
                return _written_tries(plan, jobs, ask, clean, candidates)
            except StopRequested:
                raise
            except Exception as exc:  # an overloaded model: the round goes on with the other requests' tries
                failures.append(exc)
                return {}

        groups = [hot[at:at + REVISE_GROUP] for at in range(0, len(hot), REVISE_GROUP)]
        written: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max(1, parallel)) as pool:
            for found in pool.map(request, groups):
                for key, versions in found.items():
                    written.setdefault(key, []).extend(versions)
        stats["requests_failed"] += len(failures)
        if failures and len(failures) == len(groups):
            raise failures[-1]  # nothing came back: no round to score
        before, before_local = list(parts), list(local)
        for tested, index in enumerate(hot, 1):
            tell({"round": round_, "rounds": rounds, "tested": tested - 1, "passages": len(hot), "likelihood": best,
                  "stage": "scoring"})
            passing = []
            for version in written.get(f"{index:03d} revise {round_}", []):
                new = normalize(version)[0]
                if not new or new == parts[index]:
                    stats["unchanged"] += 1
                    continue
                if facts and (problem := meaning_problem(parts[index], new, whole(parts))):
                    stats["facts"] += 1
                    stats[f"facts:{problem}"] += 1
                    continue
                trial = [*parts[:index], new, *parts[index + 1:]]
                if fits and not fits(whole(trial)):
                    stats["style"] += 1
                    continue
                passing.append(trial)
            stats["scored"] += len(passing)
            readings = score_chunks([chunk(trial, index) for trial in passing]) if passing else []
            kept = None
            for trial, reading in zip(passing, readings, strict=True):
                if reading is not None and (kept is None or (round(reading, 2), stock_phrases(trial[index]))
                                            < (round(kept[0], 2), stock_phrases(kept[1][index]))):
                    kept = (reading, trial)
            if kept and local[index] is not None and kept[0] < local[index]:
                parts[index], local[index] = kept[1][index], kept[0]
                stats["kept"] += 1
            elif kept:
                stats["not_lower"] += 1
        tell({"round": round_, "rounds": rounds, "tested": len(hot), "passages": len(hot), "likelihood": best,
              "stage": "whole"})
        now = score(whole(parts))
        if now is None or now > best:  # the whole text reads the round as worse: undo it
            parts, local = before, before_local
            stats["rounds_undone"] += 1
        else:
            best = now
        history.append(best)
        if parts == before:  # no try beat its passage: another round would likely spend its requests for nothing
            break
    return whole(parts), history, list(local)


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
    """About how long a block's new text should be: its original's length, stretched where models write short."""
    if plan.match_existing_lengths and job["previous"].strip():
        return round(count_words(job["previous"]) * getattr(plan, "stretch", 1.0))
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
        if job["kind"] == "answer_box" and not follows(plan, job):
            items.append(f"[B{number}] A new answer for the question in {job['label']}. {aim}\n"
                         f"Question to answer:\n{job['question']}\n"
                         f"Previous answer in this right-hand box (may be blank):\n{job['previous'] or '(blank)'}")
        else:
            kind = "Markdown text" if job["kind"] == "markdown_block" else "text"
            items.append(f"[B{number}] A rewrite of the existing {kind} block labelled {job['label']}. {aim}\n"
                         + (f"It answers this question:\n{job['question']}\n" if job["question"] else "")
                         + f"Existing text to regenerate:\n{original_text(plan, job)}"
                         + (f"\nThe writer's original of this part:\n{job['anchor']}" if job.get("anchor") else "")
                         + (f"\n{job['note']}" if job.get("note") else ""))
    follow = FOLLOW.get(structure(plan)) if any(follows(plan, job) for job in jobs) else None
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
    if job["kind"] == "answer_box" and not follows(plan, job):
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
            + (f"It answers this question:\n{job['question']}\n\n" if job["question"] else "")
            + f"Existing text to regenerate:\n{original_text(plan, job)}"
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
