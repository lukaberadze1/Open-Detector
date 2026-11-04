"""Attacks on finished AI text, applied after it is written and before it is revised, to see what they do to the
detector: a round trip through another language, one with no model and no quota (`free_round_trip`), and moving
clauses around. Each returns the attacked text with
what it did; what comes out is still AI text, and is logged as such with the attack in its label.

Nothing here runs a model itself: `ask` is the caller's, so the page and the benchmarks use their own connections.
It should ask without a style profile, which would pull a translation away from its source. Pure Python.
"""

from __future__ import annotations

import datetime
import email.utils
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise

from opendetector.segment import count_words, split_sentences

PIVOTS = ("German", "Japanese", "Russian", "Finnish")
ROUND_TRIP = "roundtrip"
CLAUSES = "clauses"
FREE_TRIP = "freetrip"
# Google Translate's public web endpoint: unofficial, and on 2026-11-02 its bot check answered 429 from the first
# request, so `free_round_trip` changed nothing. Identify this client honestly rather than impersonating a browser.
FREE_URL = "https://translate.googleapis.com/translate_a/single"
FREE_AGENT = "OpenDetector/1.0 (+https://github.com/lukaberadze1/Open-Detector)"
FREE_HOPS = ("en", "de", "fr", "en")  # after ClearRewriter's lib/formal-inversion.ts
FREE_CHARS = 1500  # a request's text: it travels in the address
FREE_PAUSE = 0.3  # seconds between requests
GROUP_WORDS = 400  # words a translation request carries
LENGTHS = (0.6, 1.6)  # a paragraph coming back outside these multiples of its length has lost or gained something
CLAUSE_SHARE = 0.5  # of the sentences that can be turned around

TRANSLATE = ("Translate each numbered paragraph below into {language}. Translate faithfully and completely, in natural "
             "{language}: keep every name and number, and add nothing. Keep each paragraph inside its own <p1>...</p1> "
             "tags, with the same numbers, and return only the tagged paragraphs.\n\n{paragraphs}")
_TAGGED = re.compile(r"<p(\d+)>(.*?)</p\1>", re.S)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

_CONJUNCTIONS = "because|although|though|while|since|when|if|after|before|once|unless"
_FRONTED = re.compile(rf"^({_CONJUNCTIONS}) ([^,;:]+), ([^,;:]+)([.!?])$", re.I)
_TRAILING = re.compile(rf"^([^,;:]+?) ({_CONJUNCTIONS}) ([^,;:]+)([.!?])$", re.I)
# Words that make the conjunction part of a longer phrase ("even if", "as soon as ... when"): turning those around
# breaks them.
_BOUND = {"as", "even", "only", "than", "just", "right", "long", "soon", "not", "and", "or", "but", "ever", "what"}
# A clause after one of these is what the sentence is about, not when or why it happens: "I wonder if it rains" and
# "the year when it fell" don't turn around. They may stand a word or two before the conjunction ("asked them if").
_ASKING = {"ask", "asks", "asked", "asking", "wonder", "wonders", "wondered", "wondering", "know", "knows", "knew",
           "see", "sees", "saw", "decide", "decides", "decided", "check", "checks", "checked", "tell", "tells", "told",
           "remember", "remembers", "remembered", "forget", "forgets", "forgot", "doubt", "doubts", "doubted",
           "unsure", "sure", "unclear", "clear", "understand", "understands", "understood"}
_TIMES = {"day", "days", "year", "years", "time", "times", "moment", "week", "weeks", "month", "months", "night",
          "morning", "evening", "point", "era", "period", "place", "case", "cases", "reason", "hour", "hours"}
# Only these openers can safely lose their capital: anything else may be a name.
_LOWERABLE = {"the", "a", "an", "this", "that", "these", "those", "they", "he", "she", "it", "we", "you", "there",
              "their", "his", "her", "its", "our", "my", "your", "some", "many", "most", "people", "each", "every"}
_QUOTING = re.compile("[\"“”()\\[\\]\n]")
_MIN_CLAUSE = 3  # words on each side: fewer is usually a phrase ("after dinner") rather than a clause


def attack_name(pivot: str) -> str:
    """The round trip through `pivot`, as named in `run_attacks` and in a sample's attack label."""
    return f"{ROUND_TRIP}-{pivot.lower()}"


def attack_label(attacks: Iterable[str]) -> str:
    """What the attacks add to a sample's attack label."""
    return "".join(f"+{attack}" for attack in attacks)


def _tagged(paragraphs: Iterable[tuple[int, str]]) -> str:
    return "\n\n".join(f"<p{number}>{text}</p{number}>" for number, text in paragraphs)


def _untagged(reply: str) -> dict[int, str]:
    return {int(number): text.strip() for number, text in _TAGGED.findall(reply) if text.strip()}


def _groups(paragraphs: Sequence[str], words: int) -> list[list[int]]:
    groups, size = [[]], 0
    for number, text in enumerate(paragraphs):
        length = count_words(text)
        if groups[-1] and size + length > words:
            groups.append([])
            size = 0
        groups[-1].append(number)
        size += length
    return groups


def _faithful(original: str, returned: str) -> bool:
    """Whether a paragraph came back with its numbers, about its own length."""
    low, high = LENGTHS
    return (set(_NUMBER.findall(original)) <= set(_NUMBER.findall(returned))
            and low * count_words(original) <= count_words(returned) <= high * count_words(original))


def round_trip(paragraphs: Sequence[str], ask: Callable[[str], str], pivot: str = PIVOTS[0],
               clean: Callable[[str], str] = str.strip, words: int = GROUP_WORDS) -> tuple[list[str], int]:
    """Each paragraph translated into `pivot` and back, a few paragraphs to a request (two requests a group), in
    their places. A paragraph a reply leaves out, or one that comes back without its numbers or at another length,
    stays as it was. Returns the paragraphs and how many changed."""
    result = list(paragraphs)
    changed = 0
    for group in _groups(paragraphs, words):
        worth = [number for number in group if paragraphs[number].strip()]
        if not worth:
            continue
        there = _untagged(ask(TRANSLATE.format(language=pivot, paragraphs=_tagged(
            (number, paragraphs[number]) for number in worth))))
        if not there:
            continue
        back = _untagged(ask(TRANSLATE.format(language="English", paragraphs=_tagged(sorted(there.items())))))
        for number in worth:
            returned = clean(back.get(number, ""))
            if returned and returned != paragraphs[number] and _faithful(paragraphs[number], returned):
                result[number] = returned
                changed += 1
    return result, changed


def free_translate(text: str, source: str, target: str, max_retries: int = 3) -> str:
    """Use the public endpoint, retrying 429 only when it gives a Retry-After delay.

    This endpoint is unofficial and may refuse requests. If it does, preserve the
    input rather than retrying on an invented schedule or disguising the retries.
    """
    query = urllib.parse.urlencode({"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text})
    request = urllib.request.Request(f"{FREE_URL}?{query}", headers={"User-Agent": FREE_AGENT})
    for attempt in range(max(1, max_retries)):
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                data = json.load(response)
            if not isinstance(data, list) or not data or not isinstance(data[0], list):
                return text
            return "".join(part[0] for part in data[0] if isinstance(part, list) and part and part[0])
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt + 1 >= max(1, max_retries):
                return text
            retry_after = exc.headers.get("Retry-After")
            if not retry_after:
                return text
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=datetime.UTC)
                    delay = (retry_at - datetime.datetime.now(datetime.UTC)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    return text
            # Don't retry sooner than requested; avoid blocking an evaluation for a long wait.
            if delay < 0 or delay > 30:
                return text
            time.sleep(delay)
        except (OSError, ValueError, IndexError, TypeError):
            return text
    return text


def _free_chunks(text: str) -> list[str]:
    """`text` cut at sentence ends into pieces short enough for one request."""
    chunks, current = [], ""
    for sentence in split_sentences(text):
        piece = text[sentence.start:sentence.end].strip()
        if current and len(current) + len(piece) > FREE_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip()
    return [*chunks, current] if current else []


def free_round_trip(paragraphs: Sequence[str], translate: Callable[[str, str, str], str] = free_translate,
                    clean: Callable[[str], str] = str.strip, pause: float = FREE_PAUSE) -> tuple[list[str], int]:
    """Each paragraph through FREE_HOPS, in its place, with no model and no quota. One that comes back without its
    numbers or at another length stays as it was, and so does the rest once the endpoint refuses. Returns the
    paragraphs and how many changed."""
    result, changed = list(paragraphs), 0
    for number, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        try:
            pieces = []
            for chunk in _free_chunks(paragraph):
                for source, target in pairwise(FREE_HOPS):
                    chunk = translate(chunk, source, target)
                    time.sleep(pause)
                pieces.append(chunk)
        except (OSError, ValueError, IndexError, TypeError):
            break
        returned = clean(" ".join(pieces))
        if returned and returned != paragraph and _faithful(paragraph, returned):
            result[number] = returned
            changed += 1
    return result, changed


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lowered(clause: str) -> str | None:
    """A main clause moved from the start of its sentence, or None when its first word might be a name."""
    first, _, rest = clause.partition(" ")
    if first == "I":
        return clause
    return f"{first.lower()} {rest}" if first.lower() in _LOWERABLE and first[:1].isupper() else None


def turned_around(sentence: str) -> str | None:
    """A sentence with its subordinate clause on the other side ("Because A, B." and "B because A."), or None for
    one that isn't safely of that shape. Commas elsewhere, quotations and short phrases are left alone."""
    if _QUOTING.search(sentence):
        return None
    if match := _FRONTED.match(sentence):
        conjunction, clause, main, stop = match.groups()
        if count_words(clause) < _MIN_CLAUSE or count_words(main) < _MIN_CLAUSE:
            return None
        return f"{_capitalized(main)} {conjunction.lower()} {clause}{stop}"
    if match := _TRAILING.match(sentence):
        main, conjunction, clause, stop = match.groups()
        words = [word.lower() for word in main.split()]
        if (count_words(clause) < _MIN_CLAUSE or count_words(main) < _MIN_CLAUSE
                or words[-1] in _BOUND | _TIMES or _ASKING.intersection(words[-3:])
                or re.search(rf"\b(?:{_CONJUNCTIONS})\b", main, re.I)):
            return None
        lowered = _lowered(main)
        return f"{conjunction.capitalize()} {clause}, {lowered}{stop}" if lowered else None
    return None


def reorder_clauses(text: str, share: float = CLAUSE_SHARE, seed: str = "") -> tuple[str, int]:
    """`text` with `share` of the sentences that can be turned around turned around, chosen the same way each time
    for the same text and `seed`. No words change and no request is made. Returns the text and how many changed."""
    rng = random.Random(hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest())
    pieces, at, changed = [], 0, 0
    for sentence in split_sentences(text):
        original = text[sentence.start:sentence.end]
        stripped = original.strip()
        turned = turned_around(stripped)
        if turned and rng.random() < share:
            pieces += [text[at:sentence.start], original.replace(stripped, turned)]
            at = sentence.end
            changed += 1
    return "".join(pieces) + text[at:], changed


def run_attacks(texts: Sequence[str], attacks: Iterable[str], ask: Callable[[str], str] | None = None,
                clean: Callable[[str], str] = str.strip) -> tuple[list[str], dict[str, int]]:
    """`attacks` applied in turn to the paragraphs or blocks in `texts`, each staying in its place. Returns them
    and how many each attack changed."""
    result, changes = list(texts), {}
    for attack in attacks:
        if attack == CLAUSES:
            turned = [reorder_clauses(text) for text in result]
            result, changes[attack] = [text for text, _ in turned], sum(count for _, count in turned)
        elif attack == FREE_TRIP:
            result, changes[attack] = free_round_trip(result, clean=clean)
        elif attack.startswith(f"{ROUND_TRIP}-"):
            if ask is None:
                raise ValueError(f"{attack} needs a model to ask")
            result, changes[attack] = round_trip(result, ask, attack.split("-", 1)[1].capitalize(), clean)
        else:
            raise ValueError(f"unknown attack: {attack}")
    return result, changes
