"""The math script's strongest stage, still with no text model: a white-box word search against the classifier.

Each step takes the document window the detector finds most AI-like, asks the classifier's gradient which words push it
there, gets replacements that fit both neighbours in human writing (the word-pair table) and sit close in the
classifier's own word embeddings, tests them on that window in batches, and keeps the best. Names, numbers, negations,
function words, math, citations and code never change, and at most MAX_CHANGED of the words do. `power_attack` runs the
rule stage around it. Output is labelled AI data and shows what a retrained detector should learn from. torch is
imported inside the functions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from eval.human_swaps import _like
from eval.math_attack import BIGRAM_ANY, SAFE_SYNONYMS, CachedScorer, _bigrams, fix_articles, guided_attack, sanitize
from opendetector.core.deterministic_engine import _NEGATIONS, _PROTECTED
from opendetector.normalize import normalize
from opendetector.segment import sentence_windows, split_sentences

CANDIDATES = 6  # replacements tried for each
MIN_SIMILARITY = 0.3  # cosine in the classifier's word embeddings
MIN_PAIR = 5  # times each new word pair must occur in human writing
MIN_GAIN = 0.01  # log-odds a swap must take off the window
MAX_CHANGED = 0.15  # share of the text's words
SEARCH_BUDGET = 240  # window evaluations
CHECK_EVERY = 5  # accepted swaps between whole-document scores
ROUNDS = 2
_WORD = re.compile(r"[A-Za-z][a-z]+")
_FUNCTION = frozenset(re.findall(r"\w+", (
    "the a an and or but if then than that this these those there here of in on at by for from to with without as is "
    "are was were be been being am do does did have has had will would can could may might must shall should it its he "
    "she they we you i his her their our your my me him them us who whom which what when where why how not no so very "
    "too also just only own same such both each few more most other some any all into over under again once about out "
    "up down off")))


@dataclass
class PowerResult:
    text: str
    kept: list[str] = field(default_factory=list)
    history: list[float] = field(default_factory=list)
    words_changed: int = 0


@lru_cache(maxsize=1)
def _pair_index() -> tuple[dict, dict]:
    after, before = defaultdict(dict), defaultdict(dict)
    for pair, count in _bigrams().items():
        first, _, second = pair.partition(" ")
        if count >= MIN_PAIR:
            after[first][second] = count
            before[second][first] = count
    return dict(after), dict(before)


def fitting_words(word: str, previous: str, following: str) -> list[str]:
    """Words human writers put between these neighbours, the same inflection as `word`, the best-fitting first."""
    after, before = _pair_index()
    left, right = after.get(previous, {}), before.get(following, {})
    # Only listed synonyms: words that merely fit the neighbours turned 'sang the song' into 'sang the season'.
    fits = [w for w in SAFE_SYNONYMS.get(word, ()) if (not previous or w in left) and (not following or w in right)]
    return sorted(fits, key=lambda w: -min(left.get(w, BIGRAM_ANY), right.get(w, BIGRAM_ANY)))


def _embedding(engine, word: str, cache: dict):
    if word not in cache:
        ids = engine.tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        table = engine.model.get_input_embeddings().weight if hasattr(engine.model, "get_input_embeddings") else None
        cache[word] = table[ids[0]].detach() if table is not None and len(ids) == 1 else None
    return cache[word]


def closest(engine, word: str, options: list[str], cache: dict) -> list[str]:
    """Up to CANDIDATES of `options` nearest `word` in the classifier's word embeddings."""
    import torch

    base = _embedding(engine, word, cache)
    if base is None:
        return []
    scored = []
    for option in options:
        vector = _embedding(engine, option, cache)
        if vector is not None:
            similarity = torch.nn.functional.cosine_similarity(base, vector, dim=0).item()
            if similarity >= MIN_SIMILARITY:
                scored.append((similarity, option))
    return [option for _, option in sorted(scored, reverse=True)[:CANDIDATES]]


def saliency(engine, text: str) -> list[tuple[int, int, float]]:
    """(start, end, push toward AI) for each token of `text`: gradient times embedding of the classifier's log-odds."""
    import torch

    encoded = engine.tokenizer(text, truncation=True, max_length=engine.max_length, return_offsets_mapping=True,
                               return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = encoded.to(engine.device)
    with torch.enable_grad():
        embedded = engine.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_()
        logit = engine.embedded_log_odds(embedded, encoded["attention_mask"])
        gradient = torch.autograd.grad(logit.sum(), embedded)[0]
    pushes = (gradient * embedded).sum(-1)[0].detach().float().cpu().tolist()
    return [(start, end, push) for (start, end), push in zip(offsets, pushes, strict=True) if end > start]


def word_pushes(text: str, tokens: list[tuple[int, int, float]]) -> list[tuple[float, int, int]]:
    """Editable words of `text` with the summed push of their tokens, the most AI-pushing first."""
    locked = [match.span() for _, pattern in _PROTECTED for match in pattern.finditer(text)]
    rows = []
    for match in _WORD.finditer(text):
        start, end = match.span()
        word = match.group()
        if ((word[0].isupper() and start > 0 and text[start - 2:start] not in (". ", "! ", "? ", "\n\n"))
                or word.lower() in _FUNCTION or word.lower() in _NEGATIONS
                or any(a < end and start < b for a, b in locked)):
            continue
        push = sum(value for a, b, value in tokens if a < end and start < b)
        if push > 0:
            rows.append((push, start, end))
    return sorted(rows, reverse=True)


def _neighbours(text: str, start: int, end: int) -> tuple[str, str]:
    """The words either side, or '' across punctuation: a pair the table holds doesn't cross a comma's meaning."""
    before = re.search(r"([a-z']+) $", text[max(0, start - 40):start].lower())
    after = re.match(r" ([a-z']+)", text[end:end + 40].lower())
    return before.group(1) if before else "", after.group(1) if after else ""


def gradient_search(text: str, detector, budget: int = SEARCH_BUDGET, below: float = 0.1) -> tuple[str, int]:
    """Swap words against the hottest window until the budget or the cap, keeping the version with the lowest whole-
    document score. Returns it and how many words it changed. Needs a Desklib-style engine; otherwise returns `text`."""
    engine = next((c for c in getattr(detector, "classifiers", ()) if hasattr(c, "embedded_log_odds")), None)
    if engine is None:
        return text, 0
    current = normalize(text)[0]
    cap = max(1, int(MAX_CHANGED * len(current.split())))
    cache: dict = {}
    tried: set[tuple[str, str, str]] = set()
    best_text, best_score, best_changes = current, detector.score(current), 0
    changes, spent = 0, 0
    config = detector.config

    def swap_in(chunk: str, heat: float) -> str | None:
        """The window with its best new swap in it, or None when no word of it has one left."""
        nonlocal spent
        options_of = {}
        for match in _WORD.finditer(chunk):
            previous, following = _neighbours(chunk, *match.span())
            key = (match.group().lower(), previous, following)
            if key not in tried and (previous or following) and fitting_words(*key):
                options_of[match.span()] = key
        if not options_of:  # spares the gradient, the dearest step, on a window with nothing to try
            return None
        for _, start, end in word_pushes(chunk, saliency(engine, chunk)):
            key = options_of.get((start, end))
            if key is None or key in tried:
                continue
            tried.add(key)
            options = closest(engine, key[0], fitting_words(*key), cache)
            if not options:
                continue
            word = chunk[start:end]
            trials = [fix_articles(chunk[:start] + _like(word, option) + chunk[end:]) for option in options]
            results = engine.window_logits(trials)
            spent += len(trials)
            pick = int(results.argmin())
            if heat - float(results[pick]) >= MIN_GAIN:
                return trials[pick]
            if spent >= budget:
                break
        return None

    while spent < budget and changes < cap and best_score >= below:
        windows = sentence_windows(split_sentences(current), config.doc_window_words, config.doc_stride_words)
        if not windows:
            break
        chunks = [current[w.start:w.end] for w in windows]
        logits = engine.window_logits(chunks)
        spent += len(chunks)
        for hot in logits.argsort()[::-1].tolist():  # the hottest first; a spent window hands on to the next
            swapped = swap_in(chunks[hot], float(logits[hot]))
            if swapped is not None:
                current = current[:windows[hot].start] + swapped + current[windows[hot].end:]
                changes += 1
                break
            if spent >= budget:
                break
        else:
            break  # no window has a swap left
        if swapped is None:
            break
        if changes % CHECK_EVERY == 0:
            score = detector.score(current)
            if score < best_score:
                best_text, best_score, best_changes = current, score, changes
    score = detector.score(current)
    if score < best_score:
        best_text, best_changes = current, changes
    return best_text, best_changes


def power_attack(text: str, detector, swaps: bool = True, budget: int = SEARCH_BUDGET) -> PowerResult:
    """Rules, then the word search, then the rules again on what the search left, each keeping only what lowers the
    whole-document score."""
    result = PowerResult(sanitize(text))
    scorer = CachedScorer(detector)
    for round_number in range(ROUNDS):
        result.text, kept, history = guided_attack(result.text, scorer, swaps=swaps)
        result.kept += kept
        result.history += history if not result.history else history[1:]
        searched, changed = gradient_search(result.text, detector, budget)
        if changed:
            score = detector.score(searched)
            if score < result.history[-1]:
                result.text = searched
                result.words_changed += changed
                result.kept += ["gradient"] * changed
                result.history.append(score)
                continue
        if round_number == 0 and not changed:
            break
    return result
