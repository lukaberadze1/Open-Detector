"""The no-model attack, for when the hosted quota runs out: rewrites finished AI text by rules alone, with no requests.

Passes, in order: the deterministic engine's phrase and discourse-marker rules (also at sentence starts, which its
lowercase-only rules skip), the built-in human profile's word swaps, clause reordering, and sentence splits and merges
until the spread of sentence lengths reaches a target. Math, citations and fenced code are restored exactly, and a
pass that changes a number or negation is dropped. The same text always gives the same result.

It can only attack text that already exists: it writes nothing from a prompt. Everything it saves is labelled AI:

    python -m eval.math_attack essay.txt --score
    python -m eval.math_attack samples.jsonl --score --passes rules,burst
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import re
import statistics
import unicodedata
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from eval.human_swaps import _CONTRACTIONS, _like, _tables
from eval.text_attacks import CLAUSES, run_attacks
from opendetector.config import DATA_ROOT
from opendetector.core.deterministic_engine import DeterministicEngine
from opendetector.normalize import _fix_homoglyphs, _strip_invisible
from opendetector.segment import count_words, sentence_windows, split_sentences
from opendetector.zippy import compression_readings

OUT = DATA_ROOT / "data/essays/math_redteam.jsonl"
LOGITS = Path(__file__).resolve().parent / "cache/window_logits.jsonl"  # the command line's scores, kept between runs
PASSES = ("rules", "swaps", "clauses", "burst")
TARGET_CV = 0.69  # spread of sentence lengths over their mean, as in the built-in human profile
SPLIT_ABOVE = 22  # words: a sentence this long may be cut in two
MERGE_BELOW = 12  # words: two neighbours both this short may be joined
_SPLIT = re.compile(r"(?:;\s+|,\s+(?:and|but|so)\s+)(?=(?:the|this|that|these|those|it|they|we|he|she|you|i|there)\b)",
                    re.I)
_LOWERABLE = re.compile(r"(?:The|This|That|These|Those|It|They|We|He|She|You|There|A|An)\b")


BIGRAMS = Path(__file__).resolve().parents[1] / "styles/human_default/bigrams.json.gz"
BIGRAM_ANY = 10**9  # a word at the edge of a text has no neighbour to check on that side
BIGRAM_FLOOR = 3  # a swap's rarer pair must have been seen this often
MIN_FIT = 0.05  # and at least this share as often as the original's
_TOKEN = re.compile(r"[a-z0-9']+", re.I)
MAX_LENGTH_CHANGE = 0.5  # a pass that changes a text's word count by more than this share is broken, so it's dropped
# Replacements reviewed by hand, each form listed. The ported table's synonyms hold in some senses only: 'old' became
# 'earlier' and 'point' became 'goal'.
SAFE_SYNONYMS = {
    "begin": ["start"], "begins": ["starts"], "beginning": ["starting"], "big": ["large"],
    "large": ["big"], "difficult": ["hard", "tough"], "especially": ["particularly"], "exactly": ["precisely"],
    "fast": ["quick"], "quick": ["fast"], "quickly": ["fast"], "huge": ["massive", "enormous"],
    "main": ["primary", "chief"], "often": ["frequently"], "frequently": ["often"], "really": ["truly"],
    "actually": ["really"], "basically": ["essentially"], "start": ["begin"], "starts": ["begins"],
    "starting": ["beginning"], "happen": ["occur"], "happens": ["occurs"], "happened": ["occurred"],
    "help": ["assist"], "helps": ["assists"], "helped": ["assisted"], "helping": ["assisting"],
    "necessary": ["needed", "required"], "problem": ["issue"], "problems": ["issues"], "method": ["approach"],
    "methods": ["approaches"], "try": ["attempt"], "tries": ["attempts"], "tried": ["attempted"],
    "trying": ["attempting"], "seem": ["appear"], "seems": ["appears"],
    "seemed": ["appeared"], "seeming": ["appearing"], "show": ["reveal"], "shows": ["reveals"], "showed": ["revealed"],
    "showing": ["revealing"], "result": ["outcome"], "results": ["outcomes"], "many": ["numerous"],
    "choose": ["pick"], "chooses": ["picks"], "chose": ["picked"], "get": ["obtain"], "gets": ["obtains"],
    "important": ["significant"], "significant": ["important"], "mostly": ["largely"], "usually": ["typically"],
    "typically": ["usually"], "almost": ["nearly"], "nearly": ["almost"], "correct": ["right"], "maybe": ["perhaps"],
    "perhaps": ["maybe"], "simply": ["just"], "whole": ["entire"], "entire": ["whole"], "enough": ["sufficient"],
    "buy": ["purchase"], "bought": ["purchased"],
    # Written from eval/blamed_words.py: what the detector's gradient blames, where a true equivalent exists.
    "said": ["stated"], "great": ["excellent"], "bigger": ["larger"], "generally": ["usually"],
    "absolutely": ["completely"], "completely": ["totally"], "opportunity": ["chance"], "opportunities": ["chances"],
    "currently": ["now"], "amazing": ["incredible"], "though": ["although"], "although": ["though"],
    "previously": ["earlier"], "probably": ["likely"], "enjoyed": ["liked"], "enjoy": ["like"], "enjoys": ["likes"],
    "tiny": ["small"], "answer": ["reply"], "answers": ["replies"], "notable": ["noteworthy"], "person": ["individual"],
    "particular": ["specific"], "movie": ["film"], "movies": ["films"], "film": ["movie"], "films": ["movies"],
    "buying": ["purchasing"], "children": ["kids"], "kids": ["children"], "amount": ["quantity"], "type": ["kind"],
    "types": ["kinds"], "kind": ["type"], "kinds": ["types"], "happy": ["glad"], "glad": ["happy"], "sick": ["ill"],
    "rich": ["wealthy"], "wealthy": ["rich"], "smart": ["clever"], "clever": ["smart"], "suddenly": ["abruptly"],
    "wonderful": ["lovely"], "terrible": ["awful"], "awful": ["terrible"], "beautiful": ["lovely"],
    "extremely": ["very"], "incredibly": ["very"], "certainly": ["surely"], "surely": ["certainly"],
    "various": ["several"], "several": ["various"], "around": ["about"], "approximately": ["about"],
}
# Stock phrases whose first listed replacement fails in some sentences: 'despite the fact that' to 'despite that'.
UNSAFE_PHRASES = {"the fact that", "responsible for", "the development of", "the ability to", "the results of",
                  "the implementation of", "for the purpose of", "suggests that", "indicates that", "implement",
                  "leverage", "in the first place", "in the second place", "first and foremost", "the purpose of"}
_VERB_FORMS = {"use": ("uses", "used", "using"), "help": ("helps", "helped", "helping"),
               "build": ("builds", "built", "building"), "grow": ("grows", "grew", "growing"),
               "enable": ("enables", "enabled", "enabling"), "simplify": ("simplifies", "simplified", "simplifying"),
               "show": ("shows", "showed", "showing"), "highlight": ("highlights", "highlighted", "highlighting"),
               "explain": ("explains", "explained", "explaining"), "reduce": ("reduces", "reduced", "reducing"),
               "combine": ("combines", "combined", "combining")}
_ADVERBS = {"smooth": "smoothly", "complete": "completely"}
_NO_AN = re.compile(r"(?:uni|use|usu|uti|eu|one\b|once\b|ubiq)", re.I)  # a vowel letter with a consonant's sound
_AN_ANYWAY = re.compile(r"(?:hour|honest|honou?r|heir)", re.I)
_ARTICLE = re.compile(r"\b([Aa]n?) ([A-Za-z][\w-]*)")
_REMOVED = "\ue002"
_EMOJI = r"\u2600-\u27bf\U0001f000-\U0001faff"
_EMOJI_JOIN = re.compile(rf"(?<=[{_EMOJI}])[\u200d\ufe0f](?=[\u200d{_EMOJI}]|$|\W)")
PASSAGE_SENTENCES = 1  # edits are judged a sentence at a time, so one bad swap doesn't veto a passage's good ones
BURST_SENTENCES = 3  # but splitting and joining needs neighbours
GUIDED_BUDGET = 300  # detector passes on one text
_PLAIN = {"however": "But", "nevertheless": "Still,", "nonetheless": "Still,", "therefore": "So", "consequently": "So",
          "thus": "So", "as a result": "So", "furthermore": "Also,", "moreover": "Also,", "additionally": "Also,",
          "in addition": "Also,"}
_PLAIN_START = re.compile(r"(^|(?<=[.!?]\s))(" + "|".join(_PLAIN) + r"),\s+(\S)", re.I | re.M)
GUIDED_BELOW = 0.1
GUIDED_PATIENCE = 25  # scored tries in a row with no gain before giving up on a text: a saturated one rarely moves
_MARKER_START = re.compile(
    r"(^|(?<=[.!?]\s))(?:(?:furthermore|moreover|additionally|in addition|consequently|therefore|thus|as a result|"
    r"in conclusion|in summary|overall|ultimately|indeed|notably|importantly),|it is important to note that|"
    r"it's important to note that|it is worth noting that|it should be noted that)\s+([a-z])", re.I | re.M)
# Not at a clause's end ('I know what it is') and not 'it will not', which reads as 'it won't', never 'it'll not'.
_NOTE_START = re.compile(r"^(?:it is important to note that|it's important to note that|it is worth noting that|"
                         r"it should be noted that)\s+([a-z])", re.I)
_CONTRACTION = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _CONTRACTIONS), key=len, reverse=True))
                          + r")\b(?!\s*(?:[.,;:!?\"')\]]|$))(?!\s+not\b)", re.I)


def sanitize(text: str) -> str:
    """`text` without invisible characters or look-alike letters, in NFC, with its spacing and layout left alone."""
    kept = _EMOJI_JOIN.sub(lambda m: "\ue000" if m.group() == "\u200d" else "\ue001", text)
    kept, _ = _strip_invisible(kept)
    kept, _ = _fix_homoglyphs(unicodedata.normalize("NFC", kept))
    kept = kept.replace("\r\n", "\n").replace("\r", "\n")
    return kept.replace("\ue000", "\u200d").replace("\ue001", "\ufe0f")


def fix_articles(text: str) -> str:
    """'a' or 'an' to suit the word after it, once swaps have changed that word."""
    def fix(match: re.Match) -> str:
        article, word = match.groups()
        vowel = (word[0] in "aeiouAEIOU" and not _NO_AN.match(word)) or bool(_AN_ANYWAY.match(word))
        if word.isupper() and len(word) > 1:  # initials go by their sound, which a letter can't tell
            return match.group()
        return f"{article[0]}{'n' if vowel else ''} {word}"

    return _ARTICLE.sub(fix, text)


def _inflected(found: str, target: str) -> str:
    """A lexicon replacement in the form of the word it replaces: 'showcases' becomes 'shows', not 'show'."""
    lower = found.lower()
    if target in _ADVERBS and lower.endswith("ly"):
        return _ADVERBS[target]
    if target in _VERB_FORMS:
        third, past, ing = _VERB_FORMS[target]
        if lower.endswith("ing"):
            return ing
        if lower.endswith("ed"):
            return past
        if lower.endswith("s") and not lower.endswith("ss"):
            return third
    return target


def _keeps_length(before: str, after: str) -> bool:
    words = max(1, len(before.split()))
    return abs(len(after.split()) - words) / words <= MAX_LENGTH_CHANGE or words < 8


def compression_change(before: str, after: str) -> dict[str, float]:
    """ZipPy's readings of a text before and after: higher is more AI-like, and readings only, not part of the score."""
    return {f"{name}_{when}": value for when, text in (("before", before), ("after", after))
            for name, value in compression_readings(text).items()}


def _cv(lengths: list[int]) -> float:
    return statistics.pstdev(lengths) / statistics.fmean(lengths) if len(lengths) > 1 else 0.0


def _sentence_rules(engine: DeterministicEngine, text: str) -> str:
    """The engine's rules on each sentence, its first letter lowered for them so 'Furthermore,' can match."""
    out, end = [], 0
    for span in split_sentences(text):
        out.append(text[end:span.start])
        sentence = _NOTE_START.sub(lambda m: m.group(1).upper(), text[span.start:span.end])
        new = engine.transform(sentence[:1].lower() + sentence[1:])
        out.append(new[:1].upper() + new[1:] if sentence[:1].isupper() else new)
        end = span.end
    out.append(text[end:])
    return "".join(out)


def _options(sentences: list[str]) -> list[tuple[int, list[str]]]:
    """Every allowed edit: where it starts, and what replaces that sentence (or that sentence and the next)."""
    options = []
    for i, sentence in enumerate(sentences):
        if count_words(sentence) >= SPLIT_ABOVE:
            for match in _SPLIT.finditer(sentence):
                rest = sentence[match.end():]
                options.append((i, [sentence[:match.start()].rstrip() + ".", rest[:1].upper() + rest[1:]]))
        following = sentences[i + 1] if i + 1 < len(sentences) else ""
        if (following and sentence.endswith(".") and _LOWERABLE.match(following)
                and count_words(sentence) < MERGE_BELOW and count_words(following) < MERGE_BELOW):
            options.append((i, [f"{sentence[:-1]}, and {following[:1].lower()}{following[1:]}", ""]))
    return options


def vary_lengths(text: str, target: float = TARGET_CV) -> str:
    """Split and join sentences inside paragraphs, taking the edit that widens the spread most, until `target`."""
    parts = re.split(r"(\n\s*\n)", text)
    blocks = [[part[s.start:s.end].strip() for s in split_sentences(part)]
              if i % 2 == 0 and part.strip() and "\n" not in part.strip() else None for i, part in enumerate(parts)]

    def lengths() -> list[int]:
        return [count_words(s) for block in blocks if block for s in block]

    while _cv(lengths()) < target:
        best, best_cv = None, _cv(lengths())
        for b, block in enumerate(blocks):
            for at, new in _options(block or []):
                trial = (block[:at] + [s for s in new if s] + block[at + (2 if "" in new else 1):])
                others = [count_words(s) for j, other in enumerate(blocks) if other and j != b for s in other]
                score = _cv(others + [count_words(s) for s in trial])
                if score > best_cv:
                    best, best_cv = (b, trial), score
        if best is None:
            break
        blocks[best[0]] = best[1]
    return "".join(" ".join(block) if block else part for block, part in zip(blocks, parts, strict=True))


@lru_cache(maxsize=1)
def _engine() -> DeterministicEngine:
    return DeterministicEngine()


def math_attack(text: str, passes: tuple[str, ...] = PASSES, target: float = TARGET_CV) -> tuple[str, list[str]]:
    """`text` after `passes`, and the passes that changed it."""
    engine = _engine()
    steps: dict[str, Callable[[str], str]] = {
        "rules": lambda t: _sentence_rules(engine, t),
        "swaps": fitted_swaps,
        "clauses": lambda t: run_attacks([t], (CLAUSES,), None, str)[0][0],
        "burst": lambda t: vary_lengths(t, target),
    }
    protected, tokens = engine.protect_ast_latex(sanitize(text))
    facts = engine._invariants(protected)
    applied = []
    for name in passes:
        new = steps[name](protected)
        numbers_kept = engine._invariants(new)[0] == facts[0]
        # Swaps may contract 'do not' to "don't", which is the same negation, so count negations rather than spell them.
        negations_kept = len(engine._invariants(new)[1]) == len(facts[1])
        if (new != protected and numbers_kept and negations_kept and _keeps_length(protected, new)
                and all(new.count(token) == 1 for token in tokens)):
            protected, applied = new, [*applied, name]
    return engine.restore_ast_latex(protected, tokens), applied


_SYNONYM = re.compile(r"\b(" + "|".join(sorted(SAFE_SYNONYMS, key=len, reverse=True)) + r")\b", re.I)


@lru_cache(maxsize=1)
def _bigrams() -> dict[str, int]:  # the table is read once
    if not BIGRAMS.exists():
        return {}
    with gzip.open(BIGRAMS, "rt", encoding="utf-8") as file:
        return json.load(file)


def _fit(previous: str, word: str, following: str) -> int:
    """How well `word` sits between its neighbours in human writing: its rarer pair's count, 0 if either is unseen."""
    pairs = _bigrams()
    return min(pairs.get(f"{previous} {word}", 0) if previous else BIGRAM_ANY,
               pairs.get(f"{word} {following}", 0) if following else BIGRAM_ANY)


def _capitalize_removed(text: str) -> str:
    """Take out the marks left where an opener was removed, with a capital when it opened a sentence or a quote. Only
    there, so 'e.g. the' and 'she said "yes." he left' stay as written."""
    def fix(match: re.Match) -> str:
        before = text[:match.start()].rstrip()
        opens = not before or before[-1] in ".!?\"'([\n"
        return match.group(1).upper() if opens else match.group(1)

    return re.sub(rf"{_REMOVED}+([a-z]?)", fix, text).replace(_REMOVED, "")


def fitted_swaps(text: str) -> str:
    """Stock phrases and lexicon words swapped, then every synonym that fits its neighbours in human writing at least
    as well as `MIN_FIT` of the word it replaces, the best-fitting first. Multi-word choices aren't checked inside."""
    if not text.strip():
        return text
    phrases, phrase_re, _, _, lexicon, _ = _tables()
    out = phrase_re.sub(lambda m: m.group() if m.group().lower() in UNSAFE_PHRASES
                        else _like(m.group(), phrases[m.group().lower()][0]), text)
    for pattern, target in lexicon:
        out = pattern.sub(lambda m, target=target: _like(m.group(), _inflected(m.group(), target)) if target
                          else _REMOVED, out)
    out = _capitalize_removed(out)

    def swap(match: re.Match) -> str:
        before = _TOKEN.findall(out[max(0, match.start() - 40):match.start()].lower())
        after = _TOKEN.findall(out[match.end():match.end() + 40].lower())
        previous, following = before[-1] if before else "", after[0] if after else ""
        word = match.group().lower()
        floor = max(BIGRAM_FLOOR, MIN_FIT * _fit(previous, word, following))
        fits = [(_fit(previous, choice, following), choice) for choice in SAFE_SYNONYMS[word]]
        fits = [item for item in fits if item[0] >= floor]
        return _like(match.group(), max(fits)[1]) if fits else match.group()

    out = _SYNONYM.sub(swap, out)
    out = re.sub(r"(?<=\S) {2,}(?=\S)", " ", re.sub(r"(?<=\S)[ \t]+([,.;:!?])", r"\1", out))
    return fix_articles(out)


def drop_markers(text: str) -> str:
    """Linking words at a sentence's start taken out: 'Furthermore, the plan' becomes 'The plan'."""
    def cut(match: re.Match) -> str:
        return match.group() if match.group().split()[0].isupper() else match.group(1) + match.group(2).upper()

    return _MARKER_START.sub(cut, text)


def plain_markers(text: str) -> str:
    """Linking words a sentence opens with, swapped for plain ones: rotating among 'however', 'nevertheless' and
    'nonetheless' only trades one word an AI favours for another."""
    return _PLAIN_START.sub(lambda m: m.group() if m.group(2).isupper()
                            else f"{m.group(1)}{_PLAIN[m.group(2).lower()]} {m.group(3)}", text)


def contract(text: str) -> str:
    return _CONTRACTION.sub(lambda m: m.group() if m.group().isupper()
                            else _like(m.group(), _CONTRACTIONS[m.group().lower()]), text)


def _passages(text: str, size: int = PASSAGE_SENTENCES) -> list[tuple[int, int]]:
    """Spans of `size` neighbouring sentences, never across a paragraph."""
    spans: list[tuple[int, int]] = []
    run: list = []
    for sentence in [*split_sentences(text), None]:
        if run and (sentence is None or sentence.paragraph != run[0].paragraph or len(run) == size):
            spans.append((run[0].start, run[-1].end))
            run = []
        if sentence is not None:
            run.append(sentence)
    return spans


def guided_attack(text: str, score: Callable[[str], float], budget: int = GUIDED_BUDGET,
                  below: float = GUIDED_BELOW, swaps: bool = True) -> tuple[str, list[str], list[float]]:
    """Try every edit on each passage in turn, keeping one only if the whole text then scores lower, so an edit that
    does nothing for the score (and may read worse) stays out. Stops under `below` or after `budget` scores. Returns
    the text, the edits kept and the score after each."""
    engine = _engine()
    edits: dict[str, Callable[[str], str]] = {
        "plain": plain_markers, "drop": drop_markers, "rules": lambda t: _sentence_rules(engine, t),
        "contract": contract,
        "clauses": lambda t: run_attacks([t], (CLAUSES,), None, str)[0][0],
        "burst": lambda t: vary_lengths(t, 9.0), **({"swaps": fitted_swaps} if swaps else {}),
    }
    protected, tokens = engine.protect_ast_latex(sanitize(text))
    numbers = engine._invariants(protected)[0]
    negations = len(engine._invariants(protected)[1])
    best = score(engine.restore_ast_latex(protected, tokens))
    spent, kept, history, stale = 1, [], [best], 0
    for name, edit in edits.items():
        at = 0
        while spent < budget and best >= below and stale < GUIDED_PATIENCE:
            size = BURST_SENTENCES if name == "burst" else PASSAGE_SENTENCES
            spans = [span for span in _passages(protected, size) if span[0] >= at]
            if not spans:
                break
            start, end = spans[0]
            new = edit(protected[start:end])
            at = end
            trial = protected[:start] + new + protected[end:]
            if (new == protected[start:end] or not _keeps_length(protected[start:end], new)
                    or engine._invariants(trial)[0] != numbers
                    or len(engine._invariants(trial)[1]) != negations
                    or any(trial.count(token) != 1 for token in tokens)):
                continue
            result = score(engine.restore_ast_latex(trial, tokens))
            spent += 1
            stale += 1
            if result < best:
                stale = 0
                protected, best, at = trial, result, start + len(new)
                kept.append(name)
                history.append(best)
    return engine.restore_ast_latex(protected, tokens), kept, history


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


class CachedScorer:
    """`detector.score` that remembers each window's logit. One changed sentence changes one or two document windows,
    so a search over edits needn't run the model on the rest. Only for a build of classifiers alone, whose score comes
    from window logits (slim); with Binoculars it calls `detector.score`. Padding in a batch can move a logit by a tiny
    amount, so a final figure should come from `detector.score`."""

    def __init__(self, detector, path: Path | None = None) -> None:
        self.detector = detector
        self.logits: dict[str, dict[str, float]] = {}
        self.runs = self.hits = 0
        self.path = path  # kept on disk here too, for benchmarks that see the same text again
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                name, key, value = json.loads(line)
                self.logits.setdefault(name, {})[key] = value

    def __call__(self, text: str) -> float:
        detector = self.detector
        if getattr(detector, "binoculars", None) is not None or detector.combiner is None:
            return detector.score(text)
        clean, _, sentences = detector._prepare(text)
        config = detector.config
        windows = sentence_windows(sentences, config.doc_window_words, config.doc_stride_words)
        if not windows:
            return detector.score(text)
        chunks = [clean[w.start:w.end] for w in windows]
        scores = {}
        for classifier in detector.classifiers:
            known = self.logits.setdefault(classifier.name, {})
            keys = [_key(chunk) for chunk in chunks]
            missing = list(dict.fromkeys(chunk for chunk, key in zip(chunks, keys, strict=True) if key not in known))
            self.hits += len(chunks) - len(missing)
            if missing:
                found = {_key(chunk): float(value)
                         for chunk, value in zip(missing, classifier.window_logits(missing), strict=True)}
                known.update(found)
                self.runs += len(missing)
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as file:
                        file.writelines(json.dumps([classifier.name, key, value]) + "\n"
                                        for key, value in found.items())
            scores[classifier.name] = np.array([known[key] for key in keys])
        return float(detector._probabilities([detector._doc_features(scores, windows, {})])[0])


def _read(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"text": path.read_text(encoding="utf-8")}]


def changed_passages(before: str, after: str, context: int = 3) -> list[tuple[str, str]]:
    """Each change between two texts with a few words either side, to read whether it still makes sense."""
    old, new = before.split(), after.split()
    return [(" ".join(old[max(0, i1 - context):i2 + context]), " ".join(new[max(0, j1 - context):j2 + context]))
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
            if tag != "equal"]


def documents(rows: list[dict]) -> list[dict]:
    """AI rows sharing a `group_key` joined in file order into one text, as a batch's blocks were one document. A draft
    a challenge passed over (its id ends in -draftN) is another version of the same text, so it stays out, as does a
    repeated block. Rows without a key stay as they are."""
    joined: dict[str, dict] = {}
    out: list[dict] = []
    for row in rows:
        if (row.get("label") not in (1, "1", "ai") or re.search(r"-draft\d+$", str(row.get("id", "")))
                or row.get("model") == "math-script"):  # this script's own output isn't a draft to attack
            continue
        key = row.get("group_key")
        if not key:
            out.append(row)
        elif key not in joined:
            joined[key] = {**row, "parts": [row["text"]]}
            out.append(joined[key])
        elif row["text"] not in joined[key]["parts"]:
            joined[key]["parts"].append(row["text"])
    for row in out:
        if "parts" in row:
            row["n_rows"] = len(row["parts"])
            row["text"] = "\n\n".join(row.pop("parts"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help=".txt, .md, or .jsonl with a text field")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--passes", default=",".join(PASSES))
    parser.add_argument("--target", type=float, default=TARGET_CV)
    parser.add_argument("--score", action="store_true",
                        help="guide the edits with the detector (--passes then doesn't apply) and print the scores")
    parser.add_argument("--by-group", action="store_true", help="join a .jsonl's rows by group_key first")
    parser.add_argument("--baseline", action="store_true", help="only print each text's words and score; saves nothing")
    parser.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"), help="with --score, only texts in this")
    parser.add_argument("--show-edits", action="store_true", help="print each change with the words around it")
    parser.add_argument("--budget", type=int, default=GUIDED_BUDGET, help="detector passes per text in the search")
    parser.add_argument("--profile", default="slim", choices=("slim", "full"))
    parser.add_argument("--power", action="store_true", help="with --score, add the gradient word search (slower)")
    parser.add_argument("--contextual", action="store_true",
                        help="with --score, add the model-backed word search (needs its models downloaded)")
    parser.add_argument("--paraphraser", choices=("t5", "pegasus", "dipper"),
                        help="with --contextual, rewrite sentences first")
    args = parser.parse_args()
    passes = tuple(name.strip() for name in args.passes.split(",") if name.strip())
    if unknown := set(passes) - set(PASSES):
        parser.error(f"unknown passes: {sorted(unknown)}")

    detector = None
    if args.score:
        from opendetector.config import DetectorConfig
        from opendetector.detector import Detector

        detector = Detector.from_config(DetectorConfig.for_profile(args.profile))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with args.out.open("a", encoding="utf-8") as out:
        rows = _read(args.source)
        if args.by_group:
            total = len(rows)
            rows = documents(rows)
            print(f"{total} rows joined into {len(rows)} documents, "
                  f"{sum(count_words(row['text']) >= 150 for row in rows)} of them 150+ words")
        for i, row in enumerate(rows):
            if row.get("label") in (0, "0", "human"):
                continue  # AI rewrites of human text would need their own decision; this attacks AI text only
            text = row["text"]
            if detector and args.band and not args.band[0] <= detector.score(text) <= args.band[1]:
                continue
            if detector and args.baseline:
                print(f"{i}: {count_words(text)} words, score {detector.score(text):.3f}, model {row.get('model')}, "
                      f"attack {row.get('attack')}, from {row.get('n_rows', 1)} rows", flush=True)
                continue
            if detector and count_words(text) < 150:  # the detector abstains on less, so nothing can guide the edits
                print(f"{i}: skipped, {count_words(text)} words is under the detector's 150")
                continue
            if detector:
                if args.contextual:
                    from eval.contextual_attack import MeaningGuard, contextual_attack, paraphrase_stage

                    scorer, guard = CachedScorer(detector, path=LOGITS), MeaningGuard()
                    start_text, applied = text, ["contextual"]
                    if args.paraphraser:
                        start_text = paraphrase_stage(text, scorer, guard, name=args.paraphraser).text
                        applied.append(args.paraphraser)
                    new = contextual_attack(start_text, detector, scorer, guard).text
                elif args.power:
                    from eval.gradient_attack import power_attack

                    result = power_attack(text, detector)
                    new, applied = result.text, result.kept
                else:
                    new, applied, _ = guided_attack(text, CachedScorer(detector, path=LOGITS), budget=args.budget)
                applied = sorted(set(applied))
            else:
                new, applied = math_attack(text, passes, args.target)
            if new == text:
                continue
            if args.show_edits:
                for old_part, new_part in changed_passages(text, new):
                    print(f"   - {old_part}\n   + {new_part}")
            key = row.get("group_key") or hashlib.sha256(text.encode()).hexdigest()[:16]
            record = {"text": new, "label": 1, "model": row.get("model") or "none",
                      "attack": "+".join(filter(None, (row.get("attack"), "math", *applied))),
                      "group_key": key, "source": args.source.name, **compression_change(text, new)}
            if detector:
                record["score_before"] = detector.score(text)
                record["score_after"] = detector.score(new)
                print(f"{i}: {record['score_before']:.3f} -> {record['score_after']:.3f}  {'+'.join(applied)}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
    print(f"Saved {saved} AI-labelled samples to {args.out}")


if __name__ == "__main__":
    main()
