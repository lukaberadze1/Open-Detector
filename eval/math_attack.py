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
import gzip
import hashlib
import json
import re
import statistics
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from eval.human_swaps import _CONTRACTIONS, _like, _tables
from eval.text_attacks import CLAUSES, run_attacks
from opendetector.config import DATA_ROOT
from opendetector.core.deterministic_engine import DeterministicEngine
from opendetector.segment import count_words, split_sentences

OUT = DATA_ROOT / "data/essays/math_redteam.jsonl"
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
PASSAGE_SENTENCES = 1  # edits are judged a sentence at a time, so one bad swap doesn't veto a passage's good ones
BURST_SENTENCES = 3  # but splitting and joining needs neighbours
GUIDED_BUDGET = 300  # detector passes on one text
_PLAIN = {"however": "But", "nevertheless": "Still,", "nonetheless": "Still,", "therefore": "So", "consequently": "So",
          "thus": "So", "as a result": "So", "furthermore": "Also,", "moreover": "Also,", "additionally": "Also,",
          "in addition": "Also,"}
_PLAIN_START = re.compile(r"(^|(?<=[.!?]\s))(" + "|".join(_PLAIN) + r"),\s+(\S)", re.I | re.M)
GUIDED_BELOW = 0.1
_MARKER_START = re.compile(
    r"(^|(?<=[.!?]\s))(?:furthermore|moreover|additionally|in addition|consequently|therefore|thus|as a result|"
    r"in conclusion|overall|ultimately|indeed|notably|importantly),\s+([a-z])", re.I | re.M)
_CONTRACTION = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _CONTRACTIONS), key=len, reverse=True)) + r")\b",
                          re.I)


def _cv(lengths: list[int]) -> float:
    return statistics.pstdev(lengths) / statistics.fmean(lengths) if len(lengths) > 1 else 0.0


def _sentence_rules(engine: DeterministicEngine, text: str) -> str:
    """The engine's rules on each sentence, its first letter lowered for them so 'Furthermore,' can match."""
    out, end = [], 0
    for span in split_sentences(text):
        out.append(text[end:span.start])
        sentence = text[span.start:span.end]
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
    blocks = [[text[s.start:s.end].strip() for s in split_sentences(part)]
              if i % 2 == 0 and "\n" not in part.strip() else None for i, part in enumerate(parts)]

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


def math_attack(text: str, passes: tuple[str, ...] = PASSES, target: float = TARGET_CV) -> tuple[str, list[str]]:
    """`text` after `passes`, and the passes that changed it."""
    engine = DeterministicEngine()
    steps: dict[str, Callable[[str], str]] = {
        "rules": lambda t: _sentence_rules(engine, t),
        "swaps": fitted_swaps,
        "clauses": lambda t: run_attacks([t], (CLAUSES,), None, str)[0][0],
        "burst": lambda t: vary_lengths(t, target),
    }
    protected, tokens = engine.protect_ast_latex(text)
    facts = engine._invariants(protected)
    applied = []
    for name in passes:
        new = steps[name](protected)
        numbers_kept = engine._invariants(new)[0] == facts[0]
        # Swaps may contract 'do not' to "don't", which is the same negation, so count negations rather than spell them.
        negations_kept = len(engine._invariants(new)[1]) == len(facts[1])
        if new != protected and numbers_kept and negations_kept and all(new.count(token) == 1 for token in tokens):
            protected, applied = new, [*applied, name]
    return engine.restore_ast_latex(protected, tokens), applied


@lru_cache(maxsize=1)
def _bigrams() -> dict[str, int]:
    if not BIGRAMS.exists():
        return {}
    with gzip.open(BIGRAMS, "rt", encoding="utf-8") as file:
        return json.load(file)


def _fit(previous: str, word: str, following: str) -> int:
    """How well `word` sits between its neighbours in human writing: its rarer pair's count, 0 if either is unseen."""
    pairs = _bigrams()
    return min(pairs.get(f"{previous} {word}", 0) if previous else BIGRAM_ANY,
               pairs.get(f"{word} {following}", 0) if following else BIGRAM_ANY)


def fitted_swaps(text: str) -> str:
    """Stock phrases and lexicon words swapped, then every synonym that fits its neighbours in human writing at least
    as well as `MIN_FIT` of the word it replaces, the best-fitting first. Multi-word choices aren't checked inside."""
    phrases, phrase_re, synonyms, synonym_re, lexicon, _ = _tables()
    out = phrase_re.sub(lambda m: _like(m.group(), phrases[m.group().lower()][0]), text)
    for pattern, target in lexicon:
        out = pattern.sub(lambda m, target=target: _like(m.group(), target) if target else "", out)

    def swap(match: re.Match) -> str:
        before = _TOKEN.findall(out[max(0, match.start() - 40):match.start()].lower())
        after = _TOKEN.findall(out[match.end():match.end() + 40].lower())
        previous, following = before[-1] if before else "", after[0] if after else ""
        word = match.group().lower()
        floor = max(BIGRAM_FLOOR, MIN_FIT * _fit(previous, word, following))
        fits = [(_fit(previous, choice, following), choice) for choice in synonyms[word] if " " not in choice]
        fits = [item for item in fits if item[0] >= floor]
        return _like(match.group(), max(fits)[1]) if fits else match.group()

    out = synonym_re.sub(swap, out)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"[ \t]+([,.;:!?])", r"\1", out))


def drop_markers(text: str) -> str:
    """Linking words at a sentence's start taken out: 'Furthermore, the plan' becomes 'The plan'."""
    def cut(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    return _MARKER_START.sub(cut, text)


def plain_markers(text: str) -> str:
    """Linking words a sentence opens with, swapped for plain ones: rotating among 'however', 'nevertheless' and
    'nonetheless' only trades one word an AI favours for another."""
    return _PLAIN_START.sub(lambda m: f"{m.group(1)}{_PLAIN[m.group(2).lower()]} {m.group(3)}", text)


def contract(text: str) -> str:
    return _CONTRACTION.sub(lambda m: _like(m.group(), _CONTRACTIONS[m.group().lower()]), text)


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
    engine = DeterministicEngine()
    edits: dict[str, Callable[[str], str]] = {
        "plain": plain_markers, "drop": drop_markers, "rules": lambda t: _sentence_rules(engine, t),
        "contract": contract,
        "clauses": lambda t: run_attacks([t], (CLAUSES,), None, str)[0][0],
        "burst": lambda t: vary_lengths(t, 9.0), **({"swaps": fitted_swaps} if swaps else {}),
    }
    protected, tokens = engine.protect_ast_latex(text)
    numbers = engine._invariants(protected)[0]
    negations = len(engine._invariants(protected)[1])
    best = score(engine.restore_ast_latex(protected, tokens))
    spent, kept, history = 1, [], [best]
    for name, edit in edits.items():
        at = 0
        while spent < budget and best >= below:
            size = BURST_SENTENCES if name == "burst" else PASSAGE_SENTENCES
            spans = [span for span in _passages(protected, size) if span[0] >= at]
            if not spans:
                break
            start, end = spans[0]
            new = edit(protected[start:end])
            at = end
            trial = protected[:start] + new + protected[end:]
            if (new == protected[start:end] or engine._invariants(trial)[0] != numbers
                    or len(engine._invariants(trial)[1]) != negations
                    or any(trial.count(token) != 1 for token in tokens)):
                continue
            result = score(engine.restore_ast_latex(trial, tokens))
            spent += 1
            if result < best:
                protected, best, at = trial, result, start + len(new)
                kept.append(name)
                history.append(best)
    return engine.restore_ast_latex(protected, tokens), kept, history


def _read(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"text": path.read_text(encoding="utf-8")}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help=".txt, .md, or .jsonl with a text field")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--passes", default=",".join(PASSES))
    parser.add_argument("--target", type=float, default=TARGET_CV)
    parser.add_argument("--score", action="store_true",
                        help="guide the edits with the detector (--passes then doesn't apply) and print the scores")
    parser.add_argument("--profile", default="slim", choices=("slim", "full"))
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
        for i, row in enumerate(_read(args.source)):
            if row.get("label") in (0, "0", "human"):
                continue  # AI rewrites of human text would need their own decision; this attacks AI text only
            text = row["text"]
            if detector and count_words(text) >= 150:  # the detector abstains on less, so it can't guide
                new, applied, _ = guided_attack(text, detector.score)
                applied = sorted(set(applied))
            else:
                new, applied = math_attack(text, passes, args.target)
            if new == text:
                continue
            key = row.get("group_key") or hashlib.sha256(text.encode()).hexdigest()[:16]
            record = {"text": new, "label": 1, "model": row.get("model") or "none",
                      "attack": "+".join(filter(None, (row.get("attack"), "math", *applied))),
                      "group_key": key, "source": args.source.name}
            if detector:
                record["score_before"] = detector.score(text)
                record["score_after"] = detector.score(new)
                print(f"{i}: {record['score_before']:.3f} -> {record['score_after']:.3f}  {'+'.join(applied)}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
    print(f"Saved {saved} AI-labelled samples to {args.out}")


if __name__ == "__main__":
    main()
