"""Write AI "mirrors" of human essays: same assignment, similar length, different author.

Pairing each human essay with an AI essay on the same prompt means the detector is judged
on style, not topic. Defaults to Gemini through its Interactions API, with the key
read from GEMINI_API_KEY (or the project's .env). --provider groq or --provider openrouter
writes through those, with the keys in GROQ_API_KEY and OPENROUTER_API_KEY. Any other
OpenAI-compatible endpoint works (OpenAI, a local LM Studio or Ollama server) via --base-url
and --api-key-env, so you can cover many model families. A local server needs no key and gets a 30-minute
timeout per request, since a small model on a laptop can take minutes per essay:

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 50 \\
        --base-url http://localhost:11434/v1 --model gemma3:4b --out data/essays/mirrors_gemma.jsonl

When the hosted models fail (a spent quota, a rejected key, an outage), the rest of the run
switches to a local Ollama model, gemma3:4b by default; --local-fallback '' turns that off.
Each essay's model column says which model wrote it.

Replies are unwrapped before they are saved: chat preambles ("Here's my essay!"), sign-offs
("Would you like me to...?"), --- rules and Markdown emphasis are removed.

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --style student --out data/essays/mirrors_gemini_student.jsonl

Styles: plain (default), student (a matched academic-writing condition), humanize (a detector-evasion
condition that asks the model to revise the generated sample). These are labelled benchmark conditions;
all generated samples remain labelled AI in the output data.

--style-profile adds a JSON style profile (a writer's measured rhythm, vocabulary, banned
phrases and mechanical slips) as the system prompt, so the mirrors imitate one writer:

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --style-profile data/styles/student_profile.json --out data/essays/mirrors_profile.jsonl

The profile's file name joins the attack label (plain+student_profile), so the report scores
style-imitating essays separately.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import pandas as pd

from eval import key_usage
from eval.gemini_fallback import WEB_MODEL, GeminiWebError, chat_web, web_available
from eval.human_swaps import human_swaps
from opendetector.env import load_env

HUMAN_PROFILE = "human_default_profile"  # the built-in profile, the only one word swaps run on
GEMINI_API_URL = "https://generativelanguage.googleapis.com"
GEMINI_OPENAI_URL = f"{GEMINI_API_URL}/v1beta/openai"
DEFAULT_MODEL = "gemini-3.8-flash"
# The free tier allows very few requests on the newest Flash model (20 when this was written);
# Flash-Lite's allowance is larger, so it is the fallback when the main model's quota runs out.
FALLBACK_MODEL = "gemini-3.5-flash-lite"
# The last resort once the hosted models fail: a model on a local Ollama server, with no key or quota.
LOCAL_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "gemma3:4b"
GROQ_HOST, OPENROUTER_HOST = "api.groq.com", "openrouter.ai"
# Groq's Cloudflare answers urllib's default user agent with 403 (error code 1010), whatever the key.
USER_AGENT = "OpenDetector/1.0 (+https://github.com/lukaberadze1/Open-Detector)"


@dataclass(frozen=True)
class Provider:
    """A hosted API the generators can write through, and what to start it with."""

    label: str
    base_url: str
    key_env: str
    key_page: str
    model: str
    fallbacks: tuple[str, ...]
    key_check: str = "/models"  # a path that answers only to a good key


PROVIDERS = {
    "gemini": Provider("Gemini", GEMINI_API_URL, "GEMINI_API_KEY", "https://aistudio.google.com/apikey",
                       DEFAULT_MODEL, (FALLBACK_MODEL, "gemini-3.6-flash", "gemini-3.1-flash-lite",
                                       "gemini-3-flash-preview")),
    "groq": Provider("Groq", f"https://{GROQ_HOST}/openai/v1", "GROQ_API_KEY", "https://console.groq.com/keys",
                     "openai/gpt-oss-120b", ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")),
    "openrouter": Provider("OpenRouter", f"https://{OPENROUTER_HOST}/api/v1", "OPENROUTER_API_KEY",
                           "https://openrouter.ai/keys", "cohere/north-mini-code:free",
                           ("dots-studio/dots-3-note-preview:free", "xingchen-agi/xing4.0-29b-a4b:free",
                            "google/gemma-4-31b-it:free"), key_check="/key"),
}

PROMPTS = {
    "plain": "Write an essay responding to the assignment below. Aim for about {words} words. "
             "Reply with the essay only.\n\nAssignment:\n{assignment}",
    "student": "You are a grade {grade} student. Write your essay for the assignment below in your own "
               "natural voice, about {words} words. Reply with the essay only.\n\nAssignment:\n{assignment}",
}
HUMANIZE = ("For this detector-evasion benchmark condition, revise the AI-generated sample to reduce the "
            "likelihood that an AI-text detector flags it. Preserve its claims and approximate length. "
            "This output is synthetic evaluation data and remains labelled AI. "
            "Reply with the revised sample only.\n\n{essay}")
# Told to keep a writer's slips, a model writes them everywhere: 8 times the writer's spaces before punctuation.
SLIPS_ARE_RARE = ("Slips are the exception: nearly every word, comma and full stop is spelled and spaced normally. "
                  "A slip is something the writer did now and then, not a rule. Work out how many a text this long "
                  "gets from its rate; when that comes to less than one, leave it out.")
STYLE_INTRO = ("Write in the style of the writer profiled below. Follow its rules. " + SLIPS_ARE_RARE +
               " The reference section is the measured style the rules come from.")
# For a profile with registers: one register is sent, so its habits can't leak into another.
REGISTER_INTRO = ("Write in the style of the writer profiled below, in their {register} register. Every habit, "
                  "pattern and slip has a rate measured from their own text: use each about that often, scattered "
                  "unevenly, never clustered and never in every sentence. A habit whose status is 'once' was seen a "
                  "single time: use it at most once in a long text, or not at all. " + SLIPS_ARE_RARE)
REGISTER_NAMES = {"finalized": "finished work", "notes": "quick notes"}
# Real passages show what a description can't; without the warning a model borrows their content.
EXAMPLES_INTRO = ("Real passages the writer wrote, unedited. Write like them: their sentence lengths, their few "
                  "commas, the plain words they use again and again, their slips. Never reuse their facts, names, "
                  "topics or wording.")
CHECKLIST_INTRO = "The writer's measured counts. Check the draft against them before answering:"


def load_style_profile(path: Path) -> dict:
    """A style profile as a dict. The top-level "style_profile" wrapper is optional."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data.get("style_profile", data)


def _inline(items: list) -> bool:
    """Numbers (a range) and long lists of short phrases (banned words) read best on one line;
    anything else, rules above all, stays a bulleted list."""
    if all(isinstance(x, (int, float)) for x in items):
        return True
    return len(items) >= 10 and all(isinstance(x, str) and len(x) <= 40 for x in items)


def _render(value, depth: int) -> list[str]:
    """Nested profile sections as Markdown under their JSON keys, which the rules refer to by
    name ("any word in banned_ai_cliches"). A section's plain values and inline lists come
    first, one line each, then a heading per subsection or bulleted list."""
    if isinstance(value, list):
        return [f"- {'; '.join(f'{k}: {v}' for k, v in x.items())}" if isinstance(x, dict)
                else f"- {json.dumps(x) if isinstance(x, list) else x}" for x in value]
    if not isinstance(value, dict):
        return [str(value)]
    lines, sections = [], []
    for key, item in value.items():
        if isinstance(item, list) and _inline(item):
            lines.append(f"{key}: {', '.join(map(str, item))}")
        elif isinstance(item, (dict, list)):
            sections += ["", f"{'#' * depth} {key}", *_render(item, depth + 1)]
        else:
            lines.append(f"{key}: {item}")
    return lines + sections


def profile_registers(profile: dict) -> list[str]:
    """The registers a profile describes (finished work, quick notes), first the default; [] for a
    profile in the older single-register format."""
    registers = profile.get("registers")
    return list(registers) if isinstance(registers, dict) else []


def _for_register(shared: dict, name: str, others: Iterable[str]) -> dict:
    """What every register shares, less the words tagged as another register's or as seen once. A word
    the writer uses in essays is no habit of their notes, and one seen once is no habit at all."""
    tagged = re.compile("|".join([*(rf"\({re.escape(other)}\b" for other in others), r"\(.*\bonce\b"]))
    vocabulary = [word for word in shared.get("signature_vocabulary") or [] if not tagged.search(word)]
    return {**shared, "signature_vocabulary": vocabulary} if "signature_vocabulary" in shared else shared


def style_instructions(profile: dict, register: str | None = None, examples: Sequence[str] = (),
                       checklist: Sequence[str] = ()) -> str:
    """The system prompt for a style profile. With registers: how to use the profile, the chosen
    register (the first when `register` is None or unknown) with its golden rule first, and what holds
    in every register; the other register is left out, and so is any habit seen only once. Otherwise the
    overall rule, the hard rules, then the measurements as reference; profiles without
    system_prompt_overrides are all reference.
    `examples` (the writer's own passages) and `checklist` (their measured counts) come last, from
    eval.tune_generation.style_examples."""
    if registers := profile_registers(profile):
        name = register if register in registers else registers[0]
        chosen = dict(profile["registers"][name])
        golden = chosen.pop("golden_rule", None)
        if "habits" in chosen:
            chosen["habits"] = [habit for habit in chosen["habits"] if habit.get("status") != "once"]
        lines = [REGISTER_INTRO.format(register=REGISTER_NAMES.get(name, name))]
        if golden:
            lines += ["", golden]
        if profile.get("how_to_use"):
            lines += ["", "# How to use this profile", profile["how_to_use"]]
        lines += ["", f"# The {REGISTER_NAMES.get(name, name)} register", *_render(chosen, 2)]
        if profile.get("shared"):
            others = [other for other in registers if other != name]
            lines += ["", "# In every register", *_render(_for_register(profile["shared"], name, others), 2)]
    else:
        rules = dict(profile.get("system_prompt_overrides") or {})
        golden = rules.pop("golden_transformation_rule", None)
        reference = {k: v for k, v in profile.items() if k != "system_prompt_overrides"}
        lines = [STYLE_INTRO]
        if golden:
            lines += ["", golden]
        if rules:
            lines += ["", "# Rules", *_render(rules, 2)]
        if reference:
            lines += ["", "# Reference", *_render(reference, 2)]
    if examples:
        lines += ["", "# The writer's own passages", EXAMPLES_INTRO]
        for number, passage in enumerate(examples, 1):
            lines += ["", f"Passage {number}:", passage.strip()]
    if checklist:
        lines += ["", "# Counts to check", CHECKLIST_INTRO, *(f"- {line}" for line in checklist)]
    return "\n".join(lines)


# Spacing slips a model can't keep to a rate, however it is told: (name, pattern, what replaces one).
_QUIRKS = (
    ("space_before_punctuation", re.compile(r"(?<=\w)[ \t]+(?=[.,;:!?](?:\s|$))"), ""),
    ("double_space", re.compile(r"(?<=\S) {2,}(?=\S)"), " "),
    ("lowercase_i", re.compile(r"(?<![\w'\u2019(])i(?![\w'\u2019]|\.\w)"), "I"),
)
_SPACED_BRACKET = re.compile(r"\([ \t]+([^()\n]*?)[ \t]*\)|\(([^()\n]*?)[ \t]+\)")
# Per 1,000 words, for a profile with no measurements beside it: about what the writers measured do.
DEFAULT_QUIRK_RATES = {"space_before_punctuation": 1.0, "double_space": 1.0, "lowercase_i": 0.2}


def quirk_rates(profile_path: Path, register: str | None = None) -> dict[str, float]:
    """How often the profile's writer makes each spacing slip, per 1,000 words, from the measurements.json
    beside the profile: their finished pieces for finished work, all their notes otherwise. Numbers only,
    so this holds in public mode too."""
    path = profile_path.parent / profile_path.stem.removesuffix("_profile") / "measurements.json"
    try:
        measured = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT_QUIRK_RATES)
    per_1000 = (measured.get("finalized_own" if register == "finalized" else "stacked")
                or (measured.get(register) or {}).get("whole") or {}).get("per_1000_words") or {}
    return {name: float(per_1000.get(name, rate)) for name, rate in DEFAULT_QUIRK_RATES.items()}


def _kept(text: str, spots: int, expected: float, name: str) -> set[int]:
    """Which of `spots` slips stay: `expected` of them, the fraction settled by a hash so that short
    blocks come out right on average, spread by the same hash."""
    digest = hashlib.sha256(f"{name}:{text}".encode()).digest()
    budget = int(expected) + (digest[0] / 256 < expected - int(expected))
    return set(random.Random(digest).sample(range(spots), min(budget, spots)))


def temper_quirks(text: str, rates: dict[str, float]) -> str:
    """The text with its spacing slips cut back to the writer's own rate. Only ever removes slips: a text
    with fewer than the writer's is left alone."""
    words = len(text.split())
    for name, pattern, fixed in _QUIRKS:
        keep = _kept(text, len(pattern.findall(text)), rates.get(name, 0.0) * words / 1000, name)
        spot = count()
        text = pattern.sub(lambda match, keep=keep, spot=spot, fixed=fixed:
                           match.group() if next(spot) in keep else fixed, text)
    # A spaced bracket counts two of the space_before_punctuation kind, as a writer's '( x )' would.
    keep = _kept(text, len(_SPACED_BRACKET.findall(text)),
                 rates.get("space_before_punctuation", 0.0) / 2 * words / 1000, "bracket")
    spot = count()

    def closed_up(match: re.Match) -> str:
        return match.group() if next(spot) in keep else f"({(match.group(1) or match.group(2) or '').strip()})"

    return _SPACED_BRACKET.sub(closed_up, text)


# 'collumns' for columns (8), or 'Micronutreits' and 'Macronutreits' for micronutrients and macronutrients.
_SLIP_ITEM = re.compile(r"^'([^']+)'(?: and '([^']+)')? for ([^(]+?)\s*(?:\(|$)")
DEFAULT_SLIP_SHARE = 0.1  # of a word's uses, for a slip the writing beside the profile doesn't show


def _writer_text(profile_path: Path, register: str | None) -> str:
    """The writing beside a profile, to count from: finished pieces for finished work, otherwise notes."""
    import update_style_profile as usp

    folder = profile_path.parent / profile_path.stem.removesuffix("_profile")
    if register == usp.FINALIZED and (folder / usp.FINALIZED).is_dir():
        return "\n\n".join(usp.finished_pieces(folder / usp.FINALIZED))
    corpus = folder / "corpus.jsonl"
    return "\n\n".join(entry["text"] for entry in usp.load_corpus(corpus) if "superseded" not in entry)


def writer_vocabulary(profile_path: Path) -> set[str]:
    """Every lowercase word the writer has written, in either register. Empty when the writing isn't
    beside the profile, as on a public deploy or for the built-in profile."""
    import update_style_profile as usp

    writing = _writer_text(profile_path, usp.FINALIZED) + "\n\n" + _writer_text(profile_path, None)
    return {word.lower() for word in re.findall(r"[^\W\d_]+", writing)}


def misspelling_shares(profile_path: Path, register: str | None = None) -> list[tuple[str, str, float]]:
    """The profile's recurring misspellings as (misspelling, word, share): how many of the word's uses the
    writer misspells, counted from their own writing. A share, not a rate per 1,000 words: an essay on
    Caesar names him far more than the writer's notes do, so that rate would correct nearly every use."""
    profile = load_style_profile(profile_path)
    registers = profile_registers(profile)
    if not registers:
        return []
    items = ((profile["registers"][register if register in registers else registers[0]].get("slips") or {})
             .get("recurring_misspellings") or [])
    pairs = []
    for item in items:
        if match := _SLIP_ITEM.match(item):
            first, second, right = match.groups()
            pairs += zip([first, *([second] if second else [])], right.split(" and "), strict=False)
    if not pairs:
        return []
    writing = _writer_text(profile_path, register if register in registers else registers[0])
    shares = []
    for wrong, right in pairs:
        slips = _uses(wrong, writing)
        total = slips + _uses(right, writing)
        shares.append((wrong, right, slips / total if total else DEFAULT_SLIP_SHARE))
    return shares


def _uses(word: str, text: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.I))


def temper_misspellings(text: str, shares: Iterable[tuple[str, str, float]]) -> str:
    """The text with the writer's recurring misspellings cut back to the share of uses they misspell.
    Only ever corrects: it never puts a misspelling in."""
    for wrong, right, share in shares:
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.I)
        spots = len(pattern.findall(text))
        if not spots:
            continue
        keep, spot = _kept(text, spots, share * (spots + _uses(right, text)), wrong), count()

        def corrected(match: re.Match, keep=keep, spot=spot, right=right) -> str:
            if next(spot) in keep:
                return match.group()
            return right[0].upper() + right[1:] if match.group()[0].isupper() else right

        text = pattern.sub(corrected, text)
    return text


# 'Another. piece of evidence': a sentence cut after a word or two to fit a plan, its rest in lowercase.
_CUT_SHORT = re.compile(r"(?:^|(?<=[.!?] ))([^\W\d_]+(?: [^\W\d_]+)?)\. +(?=[a-z])", re.M)


def styled_clean(profile_path: Path | None, register: str | None = None, swaps: bool = False) -> Callable[[str], str]:
    """clean_essay for samples in a style profile: also cuts spacing slips and recurring misspellings back
    to how often the writer makes them, and rejoins sentences cut off to fit a sentence plan. With `swaps`, the
    built-in human profile's replies also get eval.human_swaps; no other profile's do."""
    if profile_path is None:
        return clean_essay
    registers = profile_registers(load_style_profile(profile_path))
    register = (register if register in registers else registers[0]) if registers else register
    rates, shares = quirk_rates(profile_path, register), misspelling_shares(profile_path, register)

    swapped = human_swaps if swaps and Path(profile_path).stem == HUMAN_PROFILE else str

    def clean(reply: str) -> str:
        return swapped(temper_misspellings(temper_quirks(_CUT_SHORT.sub(r"\1 ", clean_essay(reply)), rates), shares))

    return clean


_RULE = re.compile(r"^ {0,3}([-*_])(?: *\1){2,} *$")
_HEADING = re.compile(r"^ {0,3}#{1,6} +", re.M)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
# An opening that announces the essay, or one that ends in a colon.
_PREAMBLE = re.compile(r"\b(?:here(?:['\u2019]s| is| are)|below is)\b.{0,60}?\b(?:essay|letter|draft|version|response|"
                       r"rewrite)|:$", re.I)
_SIGN_OFF = re.compile(r"^\(?(?:(?:i )?hope (?:this|that|it) helps|let me know if you|would you like me|do you want me|"
                       r"want me to|feel free to (?:ask|let me)|is there anything else|note:|word count)", re.I)


def clean_essay(reply: str) -> str:
    """The essay inside a chat reply. Models wrap it ("Okay, here's my essay!", a --- rule, a Markdown
    title, "Would you like me to change the tone?"), and a detector trained on the wrapper learns to
    spot chat replies, not AI writing. Markdown emphasis goes too: an essay pasted into a form has none.
    A short opening paragraph is only dropped when it announces the essay, and a short closing one when
    it talks to the requester, so "Here is why..." and "I hope you agree." stay."""
    blocks, current = [], []
    for line in reply.replace("\r\n", "\n").split("\n"):
        if _RULE.match(line):
            blocks.append(current)
            current = []
        else:
            current.append(line)
    blocks.append(current)
    # A rule separates the essay from the chat around it, so the essay is the longest block.
    body = "\n".join(max(blocks, key=lambda b: len(" ".join(b).split())))
    body = _ITALIC.sub(r"\2", _BOLD.sub(r"\2", _HEADING.sub("", body)))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    def chatter(p: str, pattern: re.Pattern) -> bool:
        return len(p.split()) <= 40 and bool(pattern.search(p))

    if len(paragraphs) > 1 and chatter(paragraphs[0], _PREAMBLE):
        paragraphs.pop(0)
    while len(paragraphs) > 1 and chatter(paragraphs[-1], _SIGN_OFF):
        paragraphs.pop()
    return "\n\n".join(paragraphs) or reply.strip()


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_WAIT = 120  # seconds; a server asking us to wait longer than this means a spent quota
# Seconds per request. A small model on a small machine can take minutes per essay, and a request
# that times out starts over, so a local server gets far longer than a hosted API.
REMOTE_TIMEOUT, LOCAL_TIMEOUT = 300, 1800
# Gemini says "retry in 19m15s" and Groq "try again in 1h2m3.4s" or "try again in 420ms".
_RETRY_IN = re.compile(r"(?:retry|try again) in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?")
# How long a Gemini model reasons before it answers: less is faster and spends less quota.
THINKING_LEVELS = ("minimal", "low", "medium", "high")
# The levels each model accepts, from Google's thinking guide (October 2026). A model not listed here
# is asked for the level as given, and asked again without one if it turns that level down.
_THINKING_SUPPORT = {
    "gemini-3.6-flash": THINKING_LEVELS,
    "gemini-3.5-flash-lite": THINKING_LEVELS,
    "gemini-3-flash-preview": THINKING_LEVELS,
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
    "gemini-3-pro-preview": ("low", "high"),
    "gemini-2.5-pro": ("low", "medium", "high"),
    "gemini-2.5-flash": ("low", "medium", "high"),
    "gemini-2.5-flash-lite": ("low", "medium", "high"),
}
# A laptop runs one local model, so the groups of a batch written at the same time take turns with it:
# each request gets the whole machine, and its timeout doesn't run out while it waits in Ollama's queue.
_LOCAL_TURN = threading.Lock()


class APIError(RuntimeError):
    pass


class QuotaExhausted(APIError):
    pass


class RateLimited(APIError):
    """A 429 that asks for a short wait (a per-minute limit), with the seconds it asked for, if it said."""

    def __init__(self, message: str, wait: float | None = None):
        super().__init__(message)
        self.wait = wait


_DAILY = re.compile(r"per[ -]?day", re.I)  # Gemini's "requests per day", its "PerDay" quota ids, OpenRouter's "per-day"
# Per-minute refusals in a row before a key counts as spent on a model, since a daily limit can look alike.
RATE_STRIKES = 3
MAX_HOLDS = 3  # waits for a rate-limited model's keys before one request moves on to the next model
CLOCK_SLACK = 0.05  # seconds added to a wait for a cooling key, so it has cooled when the wait ends
DEFAULT_RATE_WAIT = 30  # seconds, when a 429 doesn't say how long
RATE_MARGIN = 1  # seconds added to the wait a 429 asks for


class KeyPool:
    """Several API keys for one provider, from different accounts, each with its own allowance on each model.
    Requests go round the keys in turn, which also spreads the per-minute limits. A key out of quota on a model
    is left out for that model only, so the model is used on every other key before generate_with_fallback
    moves to the next one; a key the provider rejects is left out for every model. Shared by a batch's
    threads. Keys from one Google Cloud project share one allowance, so only keys from different projects add
    up. Never holds a key in a message: keys are named by their place in the list."""

    def __init__(self, keys: list[str] | tuple[str, ...]):
        self.keys = list(dict.fromkeys(keys)) or [""]
        self._spent: set[tuple[int, str]] = set()
        self._rejected: set[int] = set()
        self._cooling: dict[tuple[int, str], float] = {}
        self._strikes: dict[tuple[int, str], int] = {}
        self._turn: dict[str, int] = {}
        self._notes: list[str] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.keys)

    def name(self, key: str) -> str:
        return f"key {self.keys.index(key) + 1} of {len(self.keys)}"

    def _open(self, index: int, model: str) -> bool:
        return index not in self._rejected and (index, model) not in self._spent

    def next_key(self, model: str) -> str | None:
        """The next key that can ask `model` now, round the list, or None while each is spent or cooling."""
        with self._lock:
            now = time.monotonic()
            ready = [i for i in range(len(self.keys))
                     if self._open(i, model) and self._cooling.get((i, model), 0) <= now]
            if not ready:
                return None
            index = next((i for i in ready if i >= self._turn.get(model, 0)), ready[0])
            self._turn[model] = index + 1
            return self.keys[index]

    def soonest(self, model: str) -> float | None:
        """Seconds until a key cooling on `model` may ask again, or None if no key is only cooling. A single
        key never waits: one key's per-minute limit sends the request on to the next model, as it always has."""
        with self._lock:
            if len(self.keys) < 2:
                return None
            ends = [end for (i, name), end in self._cooling.items() if name == model and self._open(i, model)]
            return max(0.0, min(ends) - time.monotonic()) if ends else None

    def worked(self, key: str, model: str) -> None:
        with self._lock:
            self._strikes.pop((self.keys.index(key), model), None)

    def spend(self, key: str, model: str) -> None:
        with self._lock:
            index = self.keys.index(key)
            if (index, model) not in self._spent:
                self._spent.add((index, model))
                self._notes.append(f"{model} is out of quota on {self.name(key)}"
                                   + (", so the next key takes over." if self.open_keys(model, locked=True)
                                      else ", the last key it had."))

    def cool(self, key: str, model: str, wait: float | None) -> None:
        """A per-minute refusal. RATE_STRIKES of them in a row and the key counts as spent on the model."""
        with self._lock:
            index = self.keys.index(key)
            strikes = self._strikes[(index, model)] = self._strikes.get((index, model), 0) + 1
            wait = DEFAULT_RATE_WAIT if wait is None else wait
            self._cooling[(index, model)] = time.monotonic() + wait + RATE_MARGIN
        if strikes >= RATE_STRIKES:
            self.spend(key, model)

    def reject(self, key: str) -> None:
        with self._lock:
            index = self.keys.index(key)
            if index not in self._rejected:
                self._rejected.add(index)
                self._notes.append(f"The provider rejected {self.name(key)}; it is left out of this run.")

    def open_keys(self, model: str, locked: bool = False) -> int:
        """How many keys may still ask `model`, now or after a wait."""
        with contextlib.nullcontext() if locked else self._lock:
            return sum(self._open(i, model) for i in range(len(self.keys)))

    def usable(self) -> bool:
        """Whether any key is left that the provider hasn't rejected."""
        with self._lock:
            return len(self._rejected) < len(self.keys)

    def take_notes(self) -> list[str]:
        """What happened to the keys since the last call: one line per key spent or rejected."""
        with self._lock:
            notes, self._notes = self._notes, []
            return notes


def _key_trouble(message: str) -> bool:
    """Whether an error is about the key itself (invalid, revoked, leaked) rather than the request."""
    lower = message.lower()
    return "http 401" in lower or (("http 400" in lower or "http 403" in lower) and "key" in lower)


def _on_every_key(pool: KeyPool, model: str, ask: Callable[[str], str], pause: Callable[[float, str], None]) -> str:
    """Ask `model` with each key it still has, in turn, until one answers. A key out of quota, rate-limited or
    unable to use the model passes the request to the next key; an error that is the model's own (busy,
    timed out, unreachable) or the request's ends the model's turn, since another key wouldn't help. While
    every key left is only rate-limited, waits up to MAX_HOLDS times for the soonest, through `pause`."""
    holds, last = 0, None
    while True:
        key = pool.next_key(model)
        if key is None:
            wait = pool.soonest(model)
            if wait is None or wait > MAX_WAIT or holds == MAX_HOLDS:
                raise last or APIError(f"every key is out of quota or rate-limited on {model}")
            holds += 1
            pause(wait + CLOCK_SLACK, model)  # a coarse clock (Windows: ~15 ms) mustn't wake a hair too early
            continue
        try:
            reply = ask(key)
        except QuotaExhausted as exc:
            pool.spend(key, model)
            last = exc
        except RateLimited as exc:
            pool.cool(key, model, exc.wait)
            last = exc
        except APIError as exc:
            if _key_trouble(str(exc)):
                pool.reject(key)
            elif "HTTP 403" in str(exc) and len(pool) > 1:  # this key's project can't use the model
                pool.spend(key, model)
            else:
                raise
            last = exc
        else:
            pool.worked(key, model)
            return reply


def is_local(base_url: str) -> bool:
    return urllib.parse.urlparse(base_url).hostname in ("localhost", "127.0.0.1", "::1")


def _error_details(exc: urllib.error.HTTPError) -> tuple[str, float | None]:
    """The server's error message, and how many seconds it asks us to wait (None if it doesn't say)."""
    try:
        data = json.loads(exc.read() or b"null")
    except ValueError:
        data = None
    if isinstance(data, list) and data:  # Gemini's OpenAI-compatible endpoint wraps errors in a list
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    message = (error.get("message") if isinstance(error, dict) else error) or str(exc.reason)
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if isinstance(metadata, dict) and metadata.get("raw"):  # OpenRouter: what the model's own host said
        message = f"{message}: {metadata['raw']}"
    message = " ".join(str(message).split())

    retry_after = (exc.headers.get("retry-after") or "").strip()
    if retry_after.isdigit():
        return message, float(retry_after)
    match = _RETRY_IN.search(message)
    if match and any(match.groups()):
        hours, minutes, seconds, milliseconds = (float(g or 0) for g in match.groups())
        return message, hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return message, None


# Gemini models that turn a temperature down: they are asked at their own.
_NO_TEMPERATURE = frozenset({"gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash"})


def takes_temperature(model: str) -> bool:
    return model not in _NO_TEMPERATURE


def thinking_level(model: str, level: str | None) -> str | None:
    """The level to ask `model` for: `level` if the model has it, else the next one up it has (Gemini
    3.8 Flash starts at low), else its highest."""
    supported = _THINKING_SUPPORT.get(model)
    if level is None or supported is None or level in supported:
        return level
    return next((other for other in THINKING_LEVELS[THINKING_LEVELS.index(level):] if other in supported),
                supported[-1])


# Groq's models that reason before they answer, with how each is told to keep that out of the reply.
_GROQ_REASONING = {"openai/gpt-oss": {"include_reasoning": False}, "qwen/": {"reasoning_format": "hidden"},
                   "meta-models/muse": {"reasoning_format": "hidden"}, "minimaxai/": {"reasoning_format": "hidden"}}
_REASONING_OPTIONS = ("include_reasoning", "reasoning_format", "reasoning_effort", "reasoning")
# A model that puts its reasoning in the reply anyway. Left in, it would be saved as the sample's text.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)\s*", re.S | re.I)
_NOT_TEXT_MODELS = ("whisper", "orpheus", "guard", "content-safety")


def _provider_options(host: str | None, model: str, thinking: str | None) -> dict:
    """What Groq and OpenRouter take beyond the OpenAI request: the thinking level, and on OpenRouter which
    of a model's hosts may have the text. A writer's passages go out in these requests, so OpenRouter skips
    hosts that keep or train on prompts unless OPENROUTER_DATA_COLLECTION=allow."""
    if host == GROQ_HOST:
        hidden = next((options for prefix, options in _GROQ_REASONING.items() if model.startswith(prefix)), None)
        if hidden is None:
            return {}
        return {**hidden, **({"reasoning_effort": "low" if thinking == "minimal" else thinking} if thinking else {})}
    if host == OPENROUTER_HOST:
        collection = os.environ.get("OPENROUTER_DATA_COLLECTION", "").strip().lower() or "deny"
        return {"provider": {"data_collection": collection},
                **({"reasoning": {"effort": thinking, "exclude": True}} if thinking else {})}
    return {}


def _drop_rejected(payload: dict, message: str) -> bool:
    """Take out the options a model's 400 error turned down, to ask again without them. The prompt still
    asks for JSON, and the reply is checked either way. False if there was nothing to take out."""
    lower, dropped = message.lower(), False
    if "reasoning" in lower:
        dropped = any([payload.pop(option, None) is not None for option in _REASONING_OPTIONS])
    if any(word in lower for word in ("response_format", "json", "schema")):
        dropped = payload.pop("response_format", None) is not None or dropped
    for option in ("top_p", "top_k"):
        if option in lower:
            dropped = payload.pop(option, None) is not None or dropped
    return dropped


def _openai_usage(counts: dict) -> dict:
    """An OpenAI-style usage object under the field names Gemini's token counts use."""
    return {"total_tokens": counts.get("total_tokens") or 0, "total_input_tokens": counts.get("prompt_tokens") or 0,
            "total_output_tokens": counts.get("completion_tokens") or 0,
            "total_cached_tokens": (counts.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "total_thought_tokens": (counts.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}


OBSERVER: Callable[..., None] | None = None  # told of every request, for eval.trace; never set in public mode


def chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5,
         system: str | None = None, timeout: int = REMOTE_TIMEOUT, thinking: str | None = None,
         json_schema: dict | None = None, usage: dict | None = None, sampling: dict | None = None) -> str:
    """`_chat`, with what became of the request told to OBSERVER if one is set, and a Gemini key's request counted
    toward today's use of it (eval/key_usage.py)."""
    if urllib.parse.urlparse(base_url).hostname != "generativelanguage.googleapis.com" or not api_key:
        return _observed(base_url, api_key, model, prompt, temperature, retries, system, timeout, thinking,
                         json_schema, usage, sampling)
    try:
        reply = _observed(base_url, api_key, model, prompt, temperature, retries, system, timeout, thinking,
                          json_schema, usage, sampling)
    except QuotaExhausted as exc:
        key_usage.record(api_key, model, spent=True)
        if key_usage.enabled():  # a limit belongs to the key's project, and a public visitor brings their own
            from eval import generation_settings

            generation_settings.record_limits(generation_settings.limits_in(f"{model}: {exc}"))
        raise
    key_usage.record(api_key, model)
    return reply


def _observed(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int,
              system: str | None, timeout: int, thinking: str | None, json_schema: dict | None,
              usage: dict | None, sampling: dict | None) -> str:
    if OBSERVER is None:
        return _chat(base_url, api_key, model, prompt, temperature, retries, system, timeout, thinking, json_schema,
                     usage, sampling)
    seen: dict = {"want_thoughts": True}
    started, reply, error = time.monotonic(), None, None
    try:
        reply = _chat(base_url, api_key, model, prompt, temperature, retries, system, timeout, thinking, json_schema,
                      seen, sampling)
        return reply
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        retried = seen.pop("retry_reasons", [])
        thoughts = seen.pop("thought_summary", None)
        seen.pop("want_thoughts", None)
        if usage is not None:
            usage.update(seen)
        OBSERVER(model=model, thinking=thinking, temperature=temperature, sampling=sampling,
                 json_reply=bool(json_schema),
                 seconds=round(time.monotonic() - started, 2), retries=retried, error=error, tokens=seen,
                 thoughts=thoughts, prompt=prompt, system=system, reply=reply)


def _chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5,
          system: str | None = None, timeout: int = REMOTE_TIMEOUT, thinking: str | None = None,
          json_schema: dict | None = None, usage: dict | None = None, sampling: dict | None = None) -> str:
    """Use Google's current Interactions API for Gemini, or another provider's OpenAI API. `thinking`
    is a thinking level and `json_schema` the JSON shape to reply in, for Gemini, Groq and OpenRouter; other
    providers get their own default thinking, and only the prompt asks them for JSON. `sampling` holds top_p
    and top_k: top_k goes only to Gemini and OpenRouter, and a model that turns either down is asked again
    without it. The request's token counts (total_input_tokens, total_output_tokens and so on) are added to
    `usage`."""
    host = urllib.parse.urlparse(base_url).hostname
    if host == "generativelanguage.googleapis.com":
        return _gemini_interaction(model, api_key, prompt, temperature, retries, system, timeout,
                                   thinking_level(model, thinking), json_schema, usage, sampling)

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages, "temperature": temperature,
               **_provider_options(host, model, thinking)}
    payload.update({option: value for option, value in (sampling or {}).items()
                    if option == "top_p" or host == OPENROUTER_HOST})
    if json_schema and host in (GROQ_HOST, OPENROUTER_HOST):
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "reply", "strict": False, "schema": json_schema}}
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    attempt = 0
    while attempt < retries:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(),
                                     headers=headers)
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            if data.get("error"):  # OpenRouter answers 200 when a model's host fails after accepting the request
                error = data["error"]
                raise APIError(f"{model}: {error.get('message') if isinstance(error, dict) else error}")
            choice = (data.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                used = (data.get("usage") or {}).get("completion_tokens", "?")
                raise APIError(f"{model} ran out of room and its reply was cut off after {used} tokens")
            content = _THINK.sub("", choice.get("message", {}).get("content") or "").strip()
            if not content:
                raise APIError(f"empty response from {model}: {json.dumps(data)[:300]}")
            if usage is not None and data.get("usage"):
                usage.update(_openai_usage(data["usage"]))
            return content
        except urllib.error.HTTPError as exc:
            message, delay = _error_details(exc)
            if exc.code == 400 and _drop_rejected(payload, message):
                attempt -= 1
                continue
            # 402 is OpenRouter out of credits. Its free models' daily cap can come without a time to wait.
            if exc.code == 402 or (exc.code == 429 and ((delay is not None and delay > MAX_WAIT)
                                                        or _DAILY.search(message))):
                raise QuotaExhausted(f"{model} is out of quota: {message}") from exc
            if exc.code == 429 and attempt == retries:
                raise RateLimited(f"HTTP 429 from {base_url}: {message}", delay) from exc
            # Bad requests, unknown models and rejected keys will never succeed; don't retry them.
            if exc.code not in RETRYABLE_STATUS or attempt == retries:
                raise APIError(f"HTTP {exc.code} from {base_url}: {message}") from exc
            reason, wait = f"HTTP {exc.code}", delay if delay is not None else 2 ** (attempt - 1) * 5
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise APIError(f"could not reach {base_url}: {getattr(exc, 'reason', exc)}") from exc
            reason, wait = str(getattr(exc, "reason", exc)), 2 ** (attempt - 1) * 5
        except (http.client.HTTPException, ValueError) as exc:  # the connection dropped partway through the reply
            if attempt == retries:
                raise APIError(f"{model}'s reply was lost on the way: {exc!r}") from exc
            reason, wait = "reply lost on the way", 2 ** (attempt - 1) * 5
        print(f"  request failed ({reason}); retrying in {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _gemini_interaction(model: str, api_key: str, prompt: str, temperature: float, retries: int,
                        system: str | None, timeout: int, thinking: str | None = None,
                        json_schema: dict | None = None, usage: dict | None = None,
                        sampling: dict | None = None) -> str:
    """Create a text interaction through Google's first-party Gemini API."""
    payload = {"model": model, "input": prompt}
    if system:
        payload["system_instruction"] = system
    if json_schema:
        payload["response_format"] = {"type": "text", "mime_type": "application/json", "schema": json_schema}
    config = {}
    if takes_temperature(model):
        config["temperature"] = temperature
        config.update(sampling or {})  # a model that sets its own temperature sets these too
    if thinking:
        config["thinking_level"] = thinking
        if usage is not None and usage.get("want_thoughts"):
            config["thinking_summaries"] = "auto"
    attempt = 0
    while attempt < retries:
        if config:
            payload["generation_config"] = config
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.load(response)
            if data.get("status") not in (None, "completed"):
                error = data.get("error")
                message = error.get("message") if isinstance(error, dict) else str(error or "")
                message = message or f"interaction status was {data.get('status')}"
                raise APIError(f"{model}: {message}")
            content = "".join(
                part.get("text", "")
                for step in data.get("steps", []) if step.get("type") == "model_output"
                for part in step.get("content", []) if part.get("type") == "text"
            ).strip()
            if not content:
                raise APIError(f"empty response from {model}")
            if usage is not None:
                usage.update(data.get("usage") or {})
                if usage.get("want_thoughts"):
                    usage["thought_summary"] = _thought_summary(data)
            return content
        except urllib.error.HTTPError as exc:
            message, delay = _error_details(exc)
            lower = message.lower()
            # A model without this thinking level or JSON mode: ask again without it. The prompt still
            # asks for JSON, and the reply is checked either way.
            if exc.code == 400 and (("thinking" in lower and (config.pop("thinking_summaries", None)
                                                              or config.pop("thinking_level", None)))
                                    or ("top_p" in lower and config.pop("top_p", None))
                                    or ("top_k" in lower and config.pop("top_k", None))
                                    or (("response_format" in lower or "schema" in lower)
                                        and payload.pop("response_format", None))):
                attempt -= 1
                payload.pop("generation_config", None)
                continue
            if exc.code == 429 and ((delay is not None and delay > MAX_WAIT) or _DAILY.search(message)):
                raise QuotaExhausted(f"{model} is out of quota: {message}") from exc
            if exc.code == 429 and attempt == retries:
                raise RateLimited(f"HTTP 429 from Gemini Interactions API: {message}", delay) from exc
            if exc.code not in RETRYABLE_STATUS or attempt == retries:
                raise APIError(f"HTTP {exc.code} from Gemini Interactions API: {message}") from exc
            reason, wait = f"HTTP {exc.code}", delay if delay is not None else 2 ** (attempt - 1) * 5
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise APIError(f"could not reach Gemini Interactions API: {getattr(exc, 'reason', exc)}") from exc
            reason, wait = str(getattr(exc, "reason", exc)), 2 ** (attempt - 1) * 5
        except (http.client.HTTPException, ValueError) as exc:  # the connection dropped partway through the reply
            if attempt == retries:
                raise APIError(f"{model}'s reply was lost on the way: {exc!r}") from exc
            reason, wait = "reply lost on the way", 2 ** (attempt - 1) * 5
        print(f"  Gemini request failed ({reason}); retrying in {wait:.0f}s", file=sys.stderr)
        if usage is not None and usage.get("want_thoughts"):
            usage.setdefault("retry_reasons", []).append(reason)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _thought_summary(data: dict) -> str | None:
    """The model's own summary of its thinking, which isn't a reliable account of why it wrote what it wrote."""
    parts = [part.get("text", "") for step in data.get("steps", []) if step.get("type") == "thought"
             for part in (step.get("summary") or []) if isinstance(part, dict)]
    return "\n".join(part for part in parts if part) or None


def answer_token_limit(words: int) -> int:
    """The most tokens a local model may write for an answer of about `words` words: about twice what
    it needs, so a model that has lost track can't run on for hundreds of words."""
    return 2 * words + 200


def local_context_size(system: str | None, prompt: str, max_tokens: int) -> int:
    """A context window that holds the whole prompt and the answer. Ollama starts a model with a small
    one (4,096 tokens) and silently drops the start of a longer prompt, which is where the style
    profile and the instructions are. Counted at 3 characters a token (English runs nearer 4) and
    rounded up to a multiple of 8,192, so that the prompts of one batch share one size: a new size
    makes Ollama reload the model and lose what it has cached."""
    tokens = (len(system or "") + len(prompt)) // 3 + max_tokens + 256
    return -(-tokens // 8192) * 8192


def chat_local(local_url: str, model: str, prompt: str, temperature: float, system: str | None = None,
               max_tokens: int | None = None, context: int | None = None, timeout: int = LOCAL_TIMEOUT,
               json_schema: dict | None = None, sampling: dict | None = None) -> str:
    """Ask a local server, one request at a time. Ollama gets its own chat endpoint, which, unlike its
    OpenAI-compatible one, takes a context window, an answer length and a JSON schema. Every block of a
    batch starts with the same document, so Ollama reads it once and keeps it cached. A server without
    that endpoint (LM Studio) gets the OpenAI-compatible request."""
    with _LOCAL_TURN:
        return _chat_local(local_url, model, prompt, temperature, system, max_tokens, context, timeout, json_schema,
                           sampling)


def _chat_local(local_url: str, model: str, prompt: str, temperature: float, system: str | None,
                max_tokens: int | None, context: int | None, timeout: int, json_schema: dict | None,
                sampling: dict | None = None) -> str:
    parsed = urllib.parse.urlparse(local_url)
    max_tokens = max_tokens or answer_token_limit(500)
    options = {"temperature": temperature, "num_predict": max_tokens,
               "num_ctx": context or local_context_size(system, prompt, max_tokens), **(sampling or {})}
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if json_schema:
        body["format"] = json_schema
    req = urllib.request.Request(
        f"{parsed.scheme}://{parsed.netloc}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        message, _ = _error_details(exc)
        if exc.code == 404 and "not found" not in message.lower():  # not Ollama: no /api/chat
            return chat(local_url, "", model, prompt, temperature, retries=1, system=system, timeout=timeout,
                        sampling=sampling)
        raise APIError(f"HTTP {exc.code} from {local_url}: {message}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(f"could not reach {local_url}: {getattr(exc, 'reason', exc)}") from exc
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise APIError(f"empty response from {model}: {json.dumps(data)[:300]}")
    return content


def _sleep(seconds: float, model: str) -> None:
    print(f"  every key is rate-limited on {model}; waiting {seconds:.0f}s", file=sys.stderr)
    time.sleep(seconds)


def generate_with_fallback(base_url: str, api_key: str | KeyPool, models: list[str], prompt: str,
                           temperature: float, system: str | None, local_url: str = "", local_model: str = "",
                           max_tokens: int | None = None, local_context: int | None = None,
                           spent: set[str] | None = None, *, thinking: str | None = None,
                           json_schema: dict | None = None, timeout: int = 60,
                           sampling: dict | None = None,
                           pause: Callable[[float, str], None] = _sleep) -> tuple[str, str, list[str]]:
    """Try the hosted models in order, advancing past quota and transient failures, then the local
    model, which needs no key and has no quota. `api_key` is one key, or a KeyPool of several: then each
    model is tried on every key it has left before the next model, and while its keys are only at their
    per-minute limits the request waits for them through `pause(seconds, model)` rather than moving down.
    `max_tokens` caps the local model's answer, and
    `local_context` fixes its context window (a batch passes one size for all its blocks). Models whose
    quota is spent for hours on every key, not just busy for a moment, are added to `spent`. `thinking` and
    `json_schema` go to `chat`, as does `sampling` (top_p, top_k; to the local model too), and `timeout` is
    each hosted request's. Returns the text, the model
    that wrote it and the errors of the models that failed before it."""
    attempted = []
    errors = []
    extra = {"sampling": sampling} if sampling else {}
    pool = api_key if isinstance(api_key, KeyPool) else KeyPool([api_key])
    for model_name in dict.fromkeys(name.strip() for name in models if name.strip()):
        attempted.append(model_name)

        def ask(key: str, model_name: str = model_name) -> str:
            return chat(base_url, key, model_name, prompt, temperature, retries=1, system=system,
                        timeout=timeout, thinking=thinking, json_schema=json_schema, **extra)

        try:
            return _on_every_key(pool, model_name, ask, pause), model_name, errors
        except APIError as exc:
            errors.append(f"{model_name}: {exc}")
            if spent is not None and not pool.open_keys(model_name) and pool.usable():
                spent.add(model_name)
            message = str(exc)
            lower_message = message.lower()
            endpoint_unreachable = "could not reach" in lower_message and not any(
                word in lower_message for word in ("timed out", "timeout")
            )
            # One model's refusal says nothing about the next on Groq and OpenRouter, which host many makers'
            # models; a rejected key does. On Gemini a 400 or 403 is the key too. With several keys, only
            # running out of keys the provider takes ends the hosted models.
            routed = urllib.parse.urlparse(base_url).hostname in (GROQ_HOST, OPENROUTER_HOST)
            fatal = ("HTTP 401",) if routed else ("HTTP 400", "HTTP 401", "HTTP 403")
            if endpoint_unreachable or (any(status in message for status in fatal)
                                        and (len(pool) == 1 or not pool.usable())):
                break
    if web_available():
        attempted.append(WEB_MODEL)
        try:
            reply = chat_web(prompt, system, json_schema, model=attempted[0], thinking=thinking,
                             temperature=temperature,
                             max_tokens=max_tokens)
            return reply, WEB_MODEL, errors
        except GeminiWebError as exc:
            errors.append(f"{WEB_MODEL}: {exc}")
    if local_model:
        attempted.append(local_model)
        try:
            result = chat_local(local_url, local_model, prompt, temperature, system, max_tokens, local_context,
                                json_schema=json_schema, **extra)
            return result, local_model, errors
        except APIError as exc:
            errors.append(f"{local_model}: {exc}. Is Ollama running? Start it with `ollama serve` and "
                          f"fetch the model with `ollama pull {local_model}`.")
    details = "\n".join(f"- {error[:500]}" for error in errors) or "- No fallback models were configured."
    raise APIError(f"Every configured text model failed ({', '.join(attempted)}):\n{details}")


def unload_local_model(local_url: str, model: str, timeout: float = 30) -> bool:
    """Ask an Ollama server to drop `model` from memory now rather than after its five-minute
    keep-alive, so something else (the detector) has room. Other servers lack this endpoint;
    then nothing happens and the result is False."""
    parsed = urllib.parse.urlparse(local_url)
    req = urllib.request.Request(f"{parsed.scheme}://{parsed.netloc}/api/generate",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (OSError, ValueError):  # URLError and HTTPError are OSErrors; a malformed URL is a ValueError
        return False


def local_model_problem(local_url: str, model: str, timeout: float = 2) -> str | None:
    """Why `model` can't be used on the local server right now, or None if it can. Asks Ollama for
    its downloaded models; a server without that endpoint (LM Studio) answered, so it is running,
    and is assumed to have the model."""
    parsed = urllib.parse.urlparse(local_url)
    server = f"{parsed.scheme}://{parsed.netloc}"
    try:
        with urllib.request.urlopen(f"{server}/api/tags", timeout=timeout) as response:
            downloaded = {entry.get("name") for entry in json.load(response).get("models", [])}
    except urllib.error.HTTPError:
        return None
    except (OSError, ValueError):
        return f"nothing is answering at {server}. Start Ollama with `ollama serve`."
    if (model if ":" in model else f"{model}:latest") not in downloaded:
        return f"{model} isn't downloaded. Fetch it with `ollama pull {model}`."
    return None


def provider_for(base_url: str) -> str:
    """The name in PROVIDERS of the API at `base_url`, or "custom"."""
    host = urllib.parse.urlparse(base_url).hostname
    return next((name for name, provider in PROVIDERS.items()
                 if urllib.parse.urlparse(provider.base_url).hostname == host), "custom")


def list_models(provider: Provider, api_key: str, timeout: float = 5) -> list[str]:
    """The text models `provider` offers now, free ones first, since the free list changes often."""
    request = urllib.request.Request(f"{provider.base_url}/models",
                                     headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            entries = json.load(response)["data"]
    except (OSError, ValueError, KeyError, TypeError) as exc:  # URLError and HTTPError are OSErrors
        raise APIError(f"could not list {provider.label}'s models: {exc}") from exc
    names = [entry["id"] for entry in entries
             if entry.get("active", True) and not any(word in entry["id"].lower() for word in _NOT_TEXT_MODELS)
             and "text" in ((entry.get("architecture") or {}).get("output_modalities") or ["text"])]
    return sorted(names, key=lambda name: (not name.endswith(":free"), name))


def key_accepted(provider: Provider, api_key: str, timeout: float = 15) -> bool | None:
    """Whether `provider` takes `api_key`: None if it couldn't be asked just now."""
    request = urllib.request.Request(f"{provider.base_url}{provider.key_check}",
                                     headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        return False if exc.code in (401, 403) else None
    except OSError:
        return None


def _grade(value) -> int:
    """PERSUADE grade levels arrive as floats (8.0) when the column has blanks; default to 10."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 10


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persuade", required=True, help="PERSUADE 2.0 CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", choices=list(PROVIDERS), default="gemini",
                        help="the API to write through, and the starting value of the next options")
    parser.add_argument("--model")
    parser.add_argument("--fallback-model",
                        help="switch to this model when --model is out of quota ('' to just stop)")
    parser.add_argument("--local-fallback", default=LOCAL_MODEL,
                        help="when the hosted models fail, switch to this model on --local-url ('' to just stop)")
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--style", choices=["plain", "student", "humanize"], default="plain")
    parser.add_argument("--style-profile", type=Path,
                        help="JSON style profile to write in, sent as the system prompt; its file name is "
                             "added to the attack label (<style>+<name>)")
    parser.add_argument("--register", help="with a profile that has registers, the one to write in "
                                           "(finalized or notes; default: the profile's first)")
    parser.add_argument("--examples", type=int, default=0,
                        help="also send this many of the writer's own passages in the register, with their "
                             "measured counts (needs the profile's corpus beside it)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", help="environment variable holding the key")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    provider = PROVIDERS[args.provider]
    args.model = args.model or provider.model
    args.base_url = args.base_url or provider.base_url
    args.api_key_env = args.api_key_env or provider.key_env
    if args.fallback_model is None:
        args.fallback_model = next(iter(provider.fallbacks), "")

    if not Path(args.persuade).is_file():
        parser.error(f"{args.persuade} not found. Download PERSUADE 2.0 from "
                     "github.com/scrosseye/persuade_corpus_2.0 first.")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not is_local(args.base_url):
        parser.error(f"${args.api_key_env} is not set. Export your key for {args.base_url}, point "
                     "--api-key-env at the variable that holds it, or use a local --base-url.")
    system, attack = None, args.style
    if args.style_profile:
        examples, checklist = [], []
        try:
            if args.examples:
                from eval.tune_generation import style_examples

                examples, checklist = style_examples(args.style_profile, args.register, args.examples)
            system = style_instructions(load_style_profile(args.style_profile), args.register, examples, checklist)
        except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
            parser.error(f"could not read the style profile {args.style_profile}: {e}")
        attack = f"{args.style}+{args.style_profile.stem}" + ("+examples" if examples else "")

    df = pd.read_csv(args.persuade)
    text_col = next((c for c in ("full_text", "text") if c in df), None)
    id_col = next((c for c in ("essay_id_comp", "essay_id", "id") if c in df), None)
    if text_col is None or id_col is None:
        parser.error(f"{args.persuade} needs an essay text column (full_text or text) and an id column "
                     f"(essay_id_comp, essay_id or id); found {list(df.columns)}")
    df = df.sample(frac=1, random_state=args.seed).head(args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    template = PROMPTS["student" if args.style == "student" else "plain"]

    clean = styled_clean(args.style_profile, args.register)

    def write_essay(base_url: str, key: str, model: str, prompt: str) -> str:
        if model == WEB_MODEL:
            essay = clean(chat_web(prompt, system, temperature=args.temperature))
            if args.style == "humanize":
                essay = clean(chat_web(HUMANIZE.format(essay=essay), system, temperature=args.temperature))
            return essay
        timeout = LOCAL_TIMEOUT if is_local(base_url) else REMOTE_TIMEOUT
        essay = clean(chat(base_url, key, model, prompt, args.temperature, system=system, timeout=timeout))
        if args.style == "humanize":
            essay = clean(chat(base_url, key, model, HUMANIZE.format(essay=essay), args.temperature,
                                     system=system, timeout=timeout))
        return essay

    base_url, key, model, written = args.base_url, api_key, args.model, 0
    with out.open("a") as f:
        for _, row in df.iterrows():
            source_id = str(row[id_col])
            assignment = row.get("assignment")
            prompt = template.format(
                assignment=assignment if isinstance(assignment, str) else row.get("prompt_name", "an essay"),
                words=max(150, round(len(str(row[text_col]).split()), -1)),
                grade=_grade(row.get("grade_level")),
            )
            essay = None
            while f"{source_id}-{model}-{attack}" not in done:
                try:
                    essay = write_essay(base_url, key, model, prompt)
                    break
                except QuotaExhausted as e:
                    if args.fallback_model and model != args.fallback_model:
                        print(f"{e}\nSwitching to {args.fallback_model} for the rest of the run.", file=sys.stderr)
                        model = args.fallback_model
                        continue
                    failure = e
                except (APIError, GeminiWebError) as e:
                    failure = e
                if web_available() and model != WEB_MODEL and base_url != args.local_url:
                    print(f"{failure}\nSwitching to the Gemini website for the rest of the run.", file=sys.stderr)
                    model = WEB_MODEL
                    continue
                if not args.local_fallback or base_url == args.local_url:
                    hint = (f" Is the local server running? Start it with `ollama serve` and fetch the model with "
                            f"`ollama pull {model}`.") if base_url == args.local_url else ""
                    print(f"Stopped after {written} new essays: {failure}{hint}", file=sys.stderr)
                    return 1
                print(f"{failure}\nSwitching to {args.local_fallback} on {args.local_url} for the rest of the run.",
                      file=sys.stderr)
                base_url, key, model = args.local_url, "", args.local_fallback
            if essay is None:
                continue  # already written in an earlier run
            essay_id = f"{source_id}-{model}-{attack}"
            f.write(json.dumps({
                "id": essay_id, "text": essay, "label": 1, "dataset": "persuade_mirror",
                "domain": row.get("prompt_name", "essay"), "model": model, "attack": attack,
                "group": "ai", "group_key": source_id,
            }) + "\n")
            f.flush()
            written += 1
            print(f"  wrote {essay_id}")
    print(f"Wrote {written} new essays to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
