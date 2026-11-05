"""Public AI humanizers as an attack on the detector: data for the roadmap's Phase 2, and a test nothing here wrote.

Everything else in eval/ attacks the detector with this project's own generator. These are the tools people actually
reach for, run over AI texts of MAGE that no combiner was fitted on, and every output is saved as labelled AI data
(data/essays/humanizer_redteam.jsonl) under its source's `group_key`, so a split keeps it with its source.

    python -m eval.humanizer_attack --n 40                          # the rule-based ones: no download, no model
    python -m eval.humanizer_attack --n 40 --humanizers stealth-bart,cive-bart
    python -m eval.humanizer_attack --report-only

Humanizers (--humanizers), one model at a time so an 8 GB machine holds them:

- `academic-control`: AI-Text-Humanizer-App's default path (D. Nanjesha, MIT), bugs included: contractions expanded and
  "Moreover," and the like added. It makes text more AI-like, so a detector that loses to it has a real problem.
- `stealth-rules`: StealthHumanizer's offline fallback (R. Sarker, MIT): its phrase rules, hedges, openers and sentence
  splits. Its synonym tables stand in as eval.human_swaps, which came from the same project.
- `stealth-bart`, `cive-bart`: the two BART humanizers StealthHumanizer serves (about 1.6 GB each, safetensors only).
  Saved twice: `+sampled`, the first sample of each chunk, and `+best8`, the winner of 8 as its server picks one.
- `stealth-gemma`: its long-text path on a local Ollama model, with its prompt: `+single` and `+best4`.

`--picker` is the outside detector that picks winners, as StealthHumanizer's server does with RoBERTa; it never sees
this project's scores, so the samples aren't guided by the detector under test. Resumes from `--out`. Writes
`--report` (eval/reports/humanizers.md): the same texts before and after, by humanizer. Then add it to a fit with
`python -m eval.featurize local --path data/essays/humanizer_redteam.jsonl --out eval/cache/humanizers.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from eval.redteam_batch import meaning_kept, rewrite_measures, stock_phrases
from opendetector.config import DATA_ROOT, ROBERTA_MODEL
from opendetector.metrics import wilson_interval
from opendetector.segment import count_words, looks_like_prose, split_sentences

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_ROOT / "data/essays/humanizer_redteam.jsonl"
REPORT = ROOT / "eval/reports/humanizers.md"
BARTS = {"stealth-bart": "rudra496/stealthhumanizer-bart", "cive-bart": "cive202/humanize-ai-text-bart-large"}
RULES = ("academic-control", "stealth-rules")
CHUNK_WORDS = 120  # what these BART models are trained and served on: StealthHumanizer routes them 150 words at most
BART_SAMPLES = 8
GEMMA_TEMPERATURES = (0.85, 1.1, 1.25, 1.3)  # its ladder from a base of 0.85
LOCAL_MODEL = "gemma3:4b"


# -- AI-Text-Humanizer-App ---------------------------------------------------------------

_EXPANSIONS = (("can't", "cannot"), ("won't", "will not"), ("n't", " not"), ("'re", " are"), ("'s", " is"),
               ("'ll", " will"), ("'ve", " have"), ("'d", " would"), ("'m", " am"))
_ACADEMIC = ("Moreover,", "Additionally,", "Furthermore,", "Hence,", "Therefore,", "Consequently,", "Nonetheless,",
             "Nevertheless,")
ACADEMIC_TRANSITIONS = 0.4  # as its page sets it


def _expanded(sentence: str) -> str:
    for short, long in _EXPANSIONS:
        sentence = re.sub(re.escape(short), lambda m, long=long: long[:1].upper() + long[1:] if m.group()[0].isupper()
                          else long, sentence, flags=re.I)
    return sentence


def academic_control(text: str, rng: random.Random) -> str:
    """Its default path, turning "John's" into "John is" as it does."""
    out = []
    for sentence in split_sentences(text):
        new = _expanded(text[sentence.start:sentence.end].strip())
        out.append(f"{rng.choice(_ACADEMIC)} {new}" if rng.random() < ACADEMIC_TRANSITIONS else new)
    return " ".join(out)


# -- StealthHumanizer's rules ------------------------------------------------------------

_PHRASES = (
    (r"\bit is important to note that\b", ("notably", "it is worth noting", "the key point is")),
    (r"\bin conclusion,?", ("to wrap this up,", "overall,", "in the end,")),
    (r"\bfurthermore,?", ("also,", "just as important,", "beyond that,")),
    (r"\bmoreover,?", ("plus,", "along with that,", "also,")),
    (r"\btherefore\b", ("so", "as a result", "that is why")),
    (r"\bhowever\b", ("but", "still", "even so")),
    (r"\butilize\b", ("use",)),
    (r"\bleverage\b", ("use", "draw on", "make use of")),
    (r"\brobust\b", ("solid", "reliable", "well-tested")),
    (r"\bseamless\b", ("smooth", "low-friction", "easy to follow")),
    (r"\bdelve into\b", ("look at", "dig into", "examine")),
    (r"\btapestry\b", ("mix", "pattern", "set")),
    (r"\brealm\b", ("area", "field", "space")),
    (r"\bplays a crucial role\b", ("matters", "is central", "does important work")),
    (r"\bcomprehensive\b", ("thorough", "broad", "complete")),
    (r"\bdemonstrates\b", ("shows", "makes clear", "points to")),
    (r"\bfacilitates\b", ("helps", "makes possible", "supports")),
    (r"\boptimize\b", ("improve", "tune", "make more efficient")),
    (r"\binnovative\b", ("new", "practical", "fresh")),
    (r"\bcutting-edge\b", ("newer", "modern", "current")),
    (r"\bpivotal\b", ("key", "central", "major")),
    (r"\bintricate\b", ("complex", "detailed", "involved")),
    (r"\bmultifaceted\b", ("complex", "varied", "multi-part")),
    (r"\bsynergy\b", ("teamwork", "combined effort", "collaboration")),
    (r"\bparadigm shift\b", ("major change", "new approach", "shift in thinking")),
    (r"\bultimately\b", ("in the end", "finally", "basically")),
    (r"\bfundamentally\b", ("basically", "at its core", "essentially")),
    (r"\bunprecedented\b", ("new", "never seen before", "unique")),
    (r"\btransformative\b", ("major", "game-changing", "impactful")),
    (r"\bindispensable\b", ("vital", "necessary", "crucial")),
    (r"\belucidate\b", ("explain", "clarify", "show")),
    (r"\bunderscore\b", ("highlight", "stress", "show")),
    (r"\bmeticulous\b", ("careful", "detailed", "thorough")),
    (r"\bcognizant\b", ("aware", "mindful", "knowing")),
)
_HEDGES = ("often", "in many cases", "usually", "for many teams")
_CONNECTORS = ("In practice", "Even so", "That said", "Still", "In short", "On balance")
_ALREADY_OPENED = re.compile(r"^(In practice|Put simply|Technically|Operationally|Overall|That means|Even so|That said"
                             r"|Still|In short|On balance)", re.I)
_LONG_SENTENCE = re.compile(r"^(.{80,}?)(,\s+(?:and|but|while|which|because)\s+)(.{60,})$", re.I | re.S)


def _fnv(text: str) -> int:
    value = 2166136261
    for char in text:
        value = ((value ^ ord(char)) * 16777619) & 0xFFFFFFFF
    return value


def _rule_sentence(sentence: str, index: int) -> str:
    seed = _fnv(f"{sentence}:{index}:")
    out = sentence
    for pattern, choices in _PHRASES:
        chosen = choices[(seed + len(out)) % len(choices)]
        out = re.sub(pattern, lambda _, chosen=chosen: chosen, out, flags=re.I)

    def softened(match: re.Match) -> str:
        word = match.group().lower()
        if word == "always":
            return _HEDGES[seed % len(_HEDGES)]
        return "can support" if word == "guarantees" else "helps make"

    out = re.sub(r"\b(always|guarantees|ensures)\b", softened, out, flags=re.I)
    out = re.sub(r"\bperfect\b", "strong", out, flags=re.I)
    out = re.sub(r"\bunparalleled\b", "notable", out, flags=re.I)
    if len(out) >= 70 and index % 4 == 2 and not _ALREADY_OPENED.match(out):
        out = f"{_CONNECTORS[(seed + index) % len(_CONNECTORS)]}, {out[0].lower()}{out[1:]}"
    if len(out) >= 180 and (match := _LONG_SENTENCE.match(out)):
        first, second = match.group(1).strip().rstrip(";,:-"), match.group(3).strip()
        out = f"{first}. {second[0].upper()}{second[1:]}"
    return out


def stealth_rules(text: str, rng: random.Random) -> str:
    """Its fallback path. It decides by hashing each sentence, so `rng` goes unused."""
    from eval.human_swaps import human_swaps

    sentences = [_rule_sentence(text[s.start:s.end].strip(), number) for number, s in enumerate(split_sentences(text))]
    out = human_swaps(" ".join(sentences))
    out = re.sub(r"\s*[—]\s*", ", ", out)  # its strongest tell, it says; a range's en dash stays
    out = re.sub(r",\s*,", ",", out)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out).strip()


# -- StealthHumanizer's best-of-N --------------------------------------------------------

def chunks(text: str, words: int = CHUNK_WORDS) -> list[str]:
    """`text` as runs of whole sentences of about `words` words, as these models are served."""
    runs, current, size = [], [], 0
    for sentence in split_sentences(text):
        if current and size + sentence.words > words:
            runs.append(" ".join(current))
            current, size = [], 0
        current.append(text[sentence.start:sentence.end].strip())
        size += sentence.words
    return [*runs, " ".join(current)] if current else runs


def tidied(candidate: str) -> str:
    """StealthHumanizer's repair of a sampled output: glued words, repeated words, stray punctuation."""
    out = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", candidate)
    out = re.sub(r"\b(\w+)( \1)+\b", r"\1", out, count=3, flags=re.I)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,.;:!?]){2,}", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out + "." if out and out[-1] not in ".!?\"'" else out


def word_f1(first: str, second: str) -> float:
    one, two = first.lower().split(), second.lower().split()
    common = set(one) & set(two)
    if not common:
        return 0.0
    precision, recall = len(common) / len(one), len(common) / len(two)
    return 2 * precision * recall / (precision + recall)


def faithful(source: str, candidates: list[str]) -> list[str]:
    """The candidates to choose from: those that keep the source's meaning, else those near its words, else all."""
    return ([c for c in candidates if meaning_kept(source, c) and word_f1(source, c) >= 0.3]
            or [c for c in candidates if word_f1(source, c) >= 0.25] or candidates)


def best_by_detector(source: str, candidates: list[str], ai_probability: Callable[[list[str]], list[float]]) -> str:
    """The winner as training/TRAINING.md gives it: the lowest AI probability less half its fidelity to the source."""
    pool = faithful(source, candidates)
    scores = ai_probability(pool)
    return min(zip(pool, scores, strict=True), key=lambda pair: pair[1] - 0.5 * word_f1(pair[0], source))[0]


def style_rank(text: str) -> float:
    """The server's ranking of long text, where RoBERTa reads every candidate as about 1.0: fewer stock phrases, more
    uneven sentences, a few contractions. Lower is better."""
    words = max(1, len(text.split()))
    lengths = [count for count in (len(part.split()) for part in re.split(r"(?<=[.!?])\s+", text)) if count > 2]
    burst = statistics.pstdev(lengths) / statistics.fmean(lengths) if len(lengths) > 1 else 0.0
    contractions = len(re.findall(r"\b\w+['\u2019](?:s|t|re|ve|ll|d|m)\b", text, re.I))
    return stock_phrases(text) / words * 100 - burst - min(contractions, 4) * 0.5


# Its server's prompt for long text (deploy/main.py).
GEMMA_SYSTEM = (
    "You are an AI text humanizer. Rewrite the user's text so it reads like a real person wrote it. RULES:\n"
    "1. Preserve ALL meaning, facts, names, numbers, code, URLs, and citations.\n"
    "2. Keep the approximate length — do NOT summarize or shorten.\n"
    "3. Vary sentence length and structure (mix short and long).\n"
    "4. Use simple, everyday words and contractions (it's, don't, you're) where natural.\n"
    "5. Replace AI clichés: furthermore, moreover, additionally, delve, tapestry, navigate, complexities, in the "
    "realm of, it is important to note, a testament to, underscores, vibrant, multifaceted, unprecedented, crucial, "
    "pivotal, fostering, leveraging.\n"
    "6. Use casual transitions where they fit (but, so, then, anyway).\n"
    "7. Do NOT add personal pronouns or fillers that are not in the original: I, me, my, we, us, our, you, your, "
    "you'll, 'you know', folks. Mild spoken words (basically, really) are fine. Write about the subject, not the "
    "reader or writer.\n"
    "8. Output ONLY the rewritten text. No preamble. No explanation. No quotes around the output.")


# -- running them ------------------------------------------------------------------------

def _picker(model_id: str) -> Callable[[list[str]], list[float]]:
    """An outside detector's AI probabilities, one each."""
    import math

    from opendetector import DetectorConfig
    from opendetector.engines import build_engine

    config = DetectorConfig(roberta_model=model_id, device="cpu", dtype="float32", batch_size=2)
    engine = build_engine("roberta", config)
    return lambda texts: [1 / (1 + math.exp(-float(logit))) for logit in engine.window_logits(texts)]


def bart_writer(model_id: str, samples: int = BART_SAMPLES) -> Callable[[str], list[str]]:
    """Samples from a seq2seq humanizer, as its server draws them. Safetensors only: these are small uploaders' models,
    and a pickled checkpoint can run code when it loads."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True).eval()

    def write(text: str) -> list[str]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = model.generate(**tokens, max_length=min(1024, 2 * tokens["input_ids"].shape[1] + 32),
                                    do_sample=True, temperature=0.9, top_p=0.95,
                                    num_beams=1,  # these models default to beams
                                    num_return_sequences=samples)
        return [tidied(reply) for reply in tokenizer.batch_decode(output, skip_special_tokens=True)]
    return write


def humanize_bart(text: str, write: Callable[[str], list[str]],
                  pick: Callable[[list[str]], list[float]]) -> dict[str, str]:
    """A text chunk by chunk: the first sample of each, and the winner of each."""
    first, best = [], []
    for chunk in chunks(text):
        candidates = [reply for reply in write(chunk) if reply] or [chunk]
        first.append(candidates[0])
        best.append(best_by_detector(chunk, candidates, pick))
    return {"sampled": " ".join(first), f"best{BART_SAMPLES}": " ".join(best)}


def humanize_gemma(text: str, ask: Callable[[str, float], str],
                   pick: Callable[[list[str]], list[float]]) -> dict[str, str]:
    """A whole text, four samples up its temperature ladder: the first, and the winner by its server's ranking."""
    candidates = [tidied(ask(text, temperature)) for temperature in GEMMA_TEMPERATURES]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return {}
    pool = faithful(text, candidates)
    scores = pick(pool)
    ranked = min(zip(pool, scores, strict=True), key=lambda pair: (round(style_rank(pair[0]), 2), round(pair[1], 2)))
    return {"single": candidates[0], f"best{len(GEMMA_TEMPERATURES)}": ranked[0]}


def pick_sources(frame, n: int, used: set[str], min_words: int = 150, max_words: int = 600,
                 seed: int = 11) -> list[dict]:
    """Up to `n` AI texts of a loader's frame the detector reads, no combiner fitted on, taking each domain in turn so
    no one domain or generator fills the set."""
    rows = [row for row in frame.to_dict("records")
            if row["label"] == 1 and row["attack"] == "none" and row["id"] not in used
            and min_words <= count_words(row["text"]) <= max_words and looks_like_prose(row["text"])]
    random.Random(seed).shuffle(rows)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    chosen: list[dict] = []
    while len(chosen) < n and any(by_domain.values()):
        for domain in sorted(by_domain):
            if by_domain[domain] and len(chosen) < n:
                chosen.append(by_domain[domain].pop())
    return chosen


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def sources_for(path: Path, n: int, seed: int) -> list[dict]:
    """The texts to attack, kept beside the output so a resumed or widened run uses the same ones first."""
    saved = _load_jsonl(path)
    if len(saved) >= n:
        return saved[:n]
    from eval.datasets import load_mage

    cached = ROOT / "eval/cache/mage.jsonl"
    used = {json.loads(line)["id"] for line in cached.open(encoding="utf-8")} if cached.exists() else set()
    used |= {row["id"] for row in saved}
    frame = load_mage("test", n=20000, seed=seed)
    fresh = pick_sources(frame, n - len(saved), used, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for row in fresh:
            output.write(json.dumps({key: row[key] for key in ("id", "text", "domain", "model", "group_key")}) + "\n")
    return saved + fresh


def sample_row(source: dict, humanizer: str, variant: str, text: str, model: str) -> dict:
    """One labelled AI output. Its source is AI, so even a light edit of it is AI data: `words_retained`, not the
    `words_kept` that makes redteam_batch.copied_row read a copy as its author's own."""
    from eval.tune_revise import facts_kept

    measures = rewrite_measures(source["text"], text)
    label = f"{humanizer}+{variant}" if variant else humanizer
    return {"id": f"humanizer-{source['id']}-{label}", "text": text, "label": 1, "dataset": "humanizer_redteam",
            "domain": source.get("domain"), "model": model, "attack": f"humanizer+{label}", "humanizer": label,
            "source_id": source["id"], "source_model": source.get("model"), "group_key": source.get("group_key"),
            "words_retained": measures.get("words_kept"), "facts_retained": facts_kept(source["text"], text),
            "length_ratio": round(count_words(text) / max(1, count_words(source["text"])), 3)}


def writer_for(name: str, args) -> tuple[Callable[[str, random.Random], dict[str, str]], str, Callable[[], None]]:
    """A humanizer's function, the model to record, and what frees its memory. Models load here, once per run."""
    if name == "academic-control":
        return lambda text, rng: {"": academic_control(text, rng)}, "AI-Text-Humanizer-App", lambda: None
    if name == "stealth-rules":
        return lambda text, rng: {"": stealth_rules(text, rng)}, "StealthHumanizer-local", lambda: None
    pick = _picker(args.picker)
    if name in BARTS:
        write = bart_writer(BARTS[name], args.bart_samples)
        return lambda text, _: humanize_bart(text, write, pick), BARTS[name], _freed
    if name == "stealth-gemma":
        from eval import generate_essays

        if problem := generate_essays.local_model_problem(generate_essays.LOCAL_URL, args.local_model):
            raise RuntimeError(problem)

        def ask(text: str, temperature: float) -> str:
            return generate_essays.clean_essay(generate_essays.chat_local(
                generate_essays.LOCAL_URL, args.local_model, text, temperature, GEMMA_SYSTEM))

        def unload() -> None:
            generate_essays.unload_local_model(generate_essays.LOCAL_URL, args.local_model)
            _freed()

        return lambda text, _: humanize_gemma(text, ask, pick), args.local_model, unload
    raise ValueError(f"unknown humanizer {name!r}")


def _freed() -> None:
    from opendetector.engines import free_memory

    free_memory()


def generate(sources: list[dict], names: list[str], out: Path, args) -> int:
    """Every humanizer over every source not yet done, saved as each finishes. One that can't start is reported and
    passed over; a text it fails on is left for the next run. Returns how many of these went wrong, so a run where
    everything failed doesn't look finished."""
    problems = 0
    done = {(row["source_id"], row["humanizer"].split("+")[0]) for row in _load_jsonl(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        todo = [source for source in sources if (source["id"], name) not in done]
        if not todo:
            continue
        try:
            write, model, free = writer_for(name, args)
        except Exception as exc:  # a missing model or server shouldn't stop the others
            print(f"{name}: can't start ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
            problems += 1
            continue
        for number, source in enumerate(todo, 1):
            try:
                written = write(source["text"], random.Random(f"{args.seed}:{name}:{source['id']}"))
            except Exception as exc:  # left for the next run
                print(f"{name}, {source['id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                problems += 1
                continue
            with out.open("a", encoding="utf-8") as output:
                for variant, text in written.items():
                    output.write(json.dumps(sample_row(source, name, variant, text, model)) + "\n")
            print(f"{name}: {number}/{len(todo)}", flush=True)
        del write
        free()
    return problems


def report(rows: list[dict], sources: list[dict], profile: str) -> str:
    """The scored outputs by humanizer against the same texts before. `rows` and `sources` carry `verdict` and
    `ai_probability` (redteam_report.score)."""
    by_source = {source["id"]: source for source in sources}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("verdict") and row["source_id"] in by_source:
            groups[row["humanizer"]].append(row)
    lines = ["# Public humanizers against the detector", "",
             f"AI texts of MAGE no combiner was fitted on, through each humanizer, scored with the {profile} detector "
             "and its shipped thresholds. Written by `python -m eval.humanizer_attack`. Missed is Likely human; "
             "brackets are Wilson 95% intervals. None of these was guided by this detector.", "",
             "| Humanizer | Texts | Before % | After % | Caught | Missed | Words kept % | Names, numbers kept % "
             "| Length |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def share(group: list[dict], verdict: str) -> str:
        hits = sum(row["verdict"] == verdict for row in group)
        low, high = wilson_interval(hits, len(group))
        return f"{hits / len(group):.0%} ({low:.0%} to {high:.0%})"

    def mean(values: list[float]) -> str:
        values = [value for value in values if value is not None]
        return f"{statistics.fmean(values):.0f}" if values else "-"

    ranked = sorted(groups.items(), key=lambda item: statistics.fmean(row["ai_probability"] for row in item[1]))
    for name, group in ranked:
        before = [by_source[row["source_id"]].get("ai_probability") for row in group]
        before = [100 * value for value in before if value is not None]
        ratio = statistics.fmean(row["length_ratio"] for row in group)
        lines.append(f"| {name} | {len(group)} | {mean(before)} | "
                     f"{mean([100 * row['ai_probability'] for row in group])} | {share(group, 'likely_ai')} | "
                     f"{share(group, 'likely_human')} | {mean([row.get('words_retained') for row in group])} | "
                     f"{mean([row.get('facts_retained') for row in group])} | x{ratio:.2f} |")
    return "\n".join(lines) + "\n"


def write_report(out: Path, sources_path: Path, path: Path, profile: str, cache: Path) -> str:
    from eval.redteam_report import load_samples, score

    rows = load_samples([out])
    if not rows:
        return "No output long enough for the detector yet."
    sources = [source for source in _load_jsonl(sources_path) if count_words(source["text"]) >= 150]
    scored = score(rows + sources, profile, cache)
    text = report(scored[:len(rows)], scored[len(rows):], profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"Wrote {path}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40, help="AI texts to attack")
    parser.add_argument("--humanizers", default=",".join(RULES), help=f"from {[*RULES, *BARTS, 'stealth-gemma']}")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--report-only", action="store_true", help="only score and report what --out holds")
    parser.add_argument("--no-report", action="store_true", help="only write the samples")
    parser.add_argument("--profile", choices=("slim", "full"), default="slim")
    parser.add_argument("--picker", default=ROBERTA_MODEL, help="the outside detector that picks winners")
    parser.add_argument("--bart-samples", type=int, default=BART_SAMPLES)
    parser.add_argument("--local-model", default=LOCAL_MODEL)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    from eval.redteam_report import CACHE

    sources_path = args.out.with_suffix(".sources.jsonl")
    problems = 0
    if not args.report_only:
        names = [name.strip() for name in args.humanizers.split(",") if name.strip()]
        known = {*RULES, *BARTS, "stealth-gemma"}
        if unknown := set(names) - known:
            parser.error(f"unknown humanizers: {sorted(unknown)}")
        problems = generate(sources_for(sources_path, args.n, args.seed), names, args.out, args)
    if not args.no_report:
        print(write_report(args.out, sources_path, args.report, args.profile, CACHE))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
