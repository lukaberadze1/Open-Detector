"""The math script's model-backed stage: word swaps a language model suggests and two more models check for meaning.

The hand-reviewed synonym list keeps meaning but is small; word pairs alone are plentiful but changed it ('sang the
season'). Here a fill-in-the-blank model proposes words that fit the sentence, a sentence-similarity model and an
entailment model compare each result with the ORIGINAL sentence (so edits can't drift a step at a time), and a beam
search over the words that measurably matter keeps the versions with the lowest whole-document log-odds. A paraphraser
stage can rewrite whole sentences under the same checks: T5 on a small machine, DIPPER with a GPU.

No hosted model and no quota. Everything it writes is labelled AI by its callers. torch and transformers are imported
inside the functions, and this stays out of ui/ and APP_FILES: its models don't ship with the app.
"""

from __future__ import annotations

import math
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from eval.gradient_attack import _FUNCTION, saliency, word_pushes
from eval.human_swaps import _like
from eval.math_attack import BIGRAM_FLOOR, _engine, _fit, fix_articles
from opendetector.core.deterministic_engine import _NEGATIONS
from opendetector.normalize import normalize
from opendetector.segment import sentence_windows, split_sentences

FILL_MODEL = "distilbert/distilroberta-base"
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENTAILMENT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
PARAPHRASERS = {"t5": "humarin/chatgpt_paraphraser_on_T5_base", "dipper": "kalpeshk2011/dipper-paraphraser-xxl",
                "pegasus": "tuner007/pegasus_paraphrase"}
DIPPER_TOKENIZER = "google/t5-v1_1-large"

GRAMMAR_MODEL = "textattack/roberta-base-CoLA"
SPACY_MODEL = "en_core_web_sm"
PROPOSALS = 30  # words asked of the fill-in-the-blank model at each place
TRIED_PER_WORD = 4  # of those that pass, how many the detector scores
BEAM = 3  # versions of the text kept at each step
PATIENCE = 12  # word places in a row with no gain before a text is given up on
SHORTLIST = 30  # words the gradient names, whose real effect is then measured
BUDGET = 400  # detector passes on one text
MAX_CHANGED = 0.15  # share of the text's words
MIN_SIMILARITY = 0.85  # a swap, against the original sentence
MIN_ENTAILMENT = 0.5
# On PAWS, 0.85 and 0.7 passed 92% of true paraphrases and 76% of false ones; these pass 87% and 60%, still weak.
PARAPHRASE_SIMILARITY = 0.9
PARAPHRASE_ENTAILMENT = 0.9
MIN_ACCEPTABLE = 0.7  # the grammar model, or the original's own reading if lower, less GRAMMAR_MARGIN
GRAMMAR_MARGIN = 0.05
CONTENT = ("NOUN", "VERB", "ADJ", "ADV")
MIN_WORD_SIMILARITY = 0.65  # the swapped word itself, in counter-fitted vectors. TextFooler's 0.5 let 'days' to 'times'
MIN_PAIR_SHARE = 0.2  # a new word's rarer word pair, against the original's: 'remain on' beside 'keep on'
WORD_VECTORS = Path(__file__).resolve().parents[1] / "data/vendor/counter-fitting/word_vectors"
# Words that carry a sentence's structure, not its content: swapping them passed the sentence-level checks but broke it
# ('after' to 'now', 'around' to 'seen', 'due' to 'because').
_STRUCTURAL = frozenset([
    "after", "before", "during", "while", "since", "until", "because", "due", "around", "across", "along", "among",
    "between", "through", "within", "without", "against", "toward", "towards", "behind", "beyond", "near", "now",
    "then", "still", "yet", "even", "ever", "never", "something", "anything", "nothing", "everything", "someone",
    "anyone", "everyone", "got", "get", "gets", "went", "go", "goes", "comes", "come", "came",
])
# Light verbs and units of time: 'keep on' and 'seven days' don't survive a synonym.
_LOCKED = frozenset([
    "keep", "keeps", "kept", "give", "gives", "gave", "take", "takes", "took", "make", "makes", "made", "have", "has",
    "had", "do", "does", "did", "days", "weeks", "months", "years", "hours", "minutes", "times",
])
_IRREGULAR_PAST = frozenset([
    "told", "said", "went", "came", "got", "made", "took", "gave", "knew", "thought", "found", "left", "felt", "kept",
    "held", "brought", "began", "wrote", "saw", "ran", "won", "lost", "paid", "sold", "sent", "stood", "understood",
    "became",
])
_TOKENS = re.compile(r"([A-Za-z]+)")
_FORMS = ("ing", "ed", "ly", "s")


def log_odds(p: float) -> float:
    p = min(max(p, 1e-15), 1 - 1e-15)
    return math.log(p / (1 - p))


@lru_cache(maxsize=1)
def _word_vectors() -> dict[str, np.ndarray]:
    """Vectors tuned so synonyms sit close and antonyms apart, which ordinary ones don't: 'comes' and 'goes'."""
    with zipfile.ZipFile(WORD_VECTORS / "counter-fitted-vectors.txt.zip") as archive:
        name = archive.namelist()[0]
        with archive.open(name) as file:
            rows = (line.decode("utf-8", "ignore").split(" ", 1) for line in file)
            return {word: np.fromstring(rest, sep=" ", dtype=np.float32) for word, rest in rows}


@lru_cache(maxsize=1)
def _antonyms() -> frozenset[frozenset[str]]:
    """Word pairs listed as opposites, read from the counter-fitting project's lists when they're there."""
    pairs = set()
    for name in ("wordnet_antonyms.txt", "ppdb_antonyms.txt"):
        path = WORD_VECTORS.parent / "linguistic_constraints" / name
        if path.exists():
            pairs.update(frozenset(line.split()) for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(pairs)


def available() -> bool:
    """Whether this machine has the stage's word vectors. The models themselves come from the Hugging Face cache."""
    return (WORD_VECTORS / "counter-fitted-vectors.txt.zip").exists()


def close_word(old: str, new: str) -> bool:
    """Whether two words mean nearly the same. A word the vectors don't know fails: it can't be vouched for."""
    if frozenset((old.lower(), new.lower())) in _antonyms():
        return False
    vectors = _word_vectors()
    a, b = vectors.get(old.lower()), vectors.get(new.lower())
    if a is None or b is None:
        return False
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) >= MIN_WORD_SIMILARITY


def same_form(old: str, new: str) -> bool:
    """A rough inflection check, so a past tense isn't swapped for a plural or a present ('told' to 'tell')."""
    old, new = old.lower(), new.lower()
    if (old in _IRREGULAR_PAST or old.endswith("ed")) != (new in _IRREGULAR_PAST or new.endswith("ed")):
        return False
    ending = lambda word: next((end for end in _FORMS if word.endswith(end)), "")  # noqa: E731
    return (ending(old) == ending(new) or {ending(old), ending(new)} == {"ed", ""})


def allowed(old: str, new: str, sentence: str, check_form: bool = True) -> bool:
    new_lower = new.lower()
    if old.lower() in _LOCKED or new_lower in _LOCKED:  # light verbs and units of time: 'keep on' , 'seven days'
        return False
    return (new.isalpha() and len(new) > 2 and new_lower != old.lower() and new_lower not in _FUNCTION
            and new_lower not in _NEGATIONS and old.lower() not in _STRUCTURAL
            and new_lower not in _STRUCTURAL and (not check_form or same_form(old, new))
            and not re.search(rf"\b{re.escape(new_lower)}\b", sentence.lower()))


class Tokens:
    """A text as words and what's between them. A swap changes one entry, so places stay valid in every version."""

    def __init__(self, text: str) -> None:
        self.parts = _TOKENS.split(text)
        self.starts: dict[int, int] = {}
        at = 0
        for i, part in enumerate(self.parts):
            if i % 2:
                self.starts[at] = i
            at += len(part)
        spans = [(s.start, s.end) for s in split_sentences(text)]
        self.sentence_of: dict[int, tuple[int, int]] = {}
        for start, end in spans:
            inside = [i for at, i in self.starts.items() if start <= at < end]
            for i in inside:
                self.sentence_of[i] = (inside[0], inside[-1])

    def editable(self, i: int) -> bool:
        """Not part of a contraction or a hyphenated word."""
        return (i in self.sentence_of and self.parts[i - 1][-1:] not in ("'", "\u2019", "-")
                and self.parts[i + 1][:1] not in ("'", "\u2019", "-"))


def join(parts: list[str]) -> str:
    return "".join(parts)


def swapped(parts: list[str], i: int, word: str) -> list[str]:
    new = list(parts)
    new[i] = _like(parts[i], word)
    if i >= 2 and parts[i - 2].lower() in ("a", "an") and parts[i - 1] == " ":  # 'a' or 'an' goes with the new word
        new[i - 2] = fix_articles(f"{parts[i - 2]} {new[i]}").split(" ")[0]
    return new


def sentence_text(parts: list[str], bounds: tuple[int, int]) -> str:
    return join(parts[bounds[0]:bounds[1] + 1])


def neighbours(parts: list[str], i: int) -> tuple[str, str]:
    """The words either side when only a space separates them, for the word-pair table."""
    before = parts[i - 2].lower() if i >= 2 and parts[i - 1] == " " else ""
    after = parts[i + 2].lower() if i + 2 < len(parts) and parts[i + 1] == " " else ""
    return before, after


@lru_cache(maxsize=1)
def _fill_model():
    from transformers import pipeline

    return pipeline("fill-mask", model=FILL_MODEL, device="cpu", top_k=PROPOSALS)


@lru_cache(maxsize=1)
def _listed_synonyms() -> dict[str, tuple[str, ...]]:
    """The counter-fitting project's PPDB synonym pairs, both ways, when they're there."""
    path = WORD_VECTORS.parent / "linguistic_constraints/ppdb_synonyms.txt"
    pairs: dict[str, set[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            words = line.split()
            if len(words) == 2:
                pairs.setdefault(words[0], set()).add(words[1])
                pairs.setdefault(words[1], set()).add(words[0])
    return {word: tuple(sorted(others)) for word, others in pairs.items()}


def propose(parts: list[str], i: int, bounds: tuple[int, int]) -> list[str]:
    """Words the fill-in-the-blank model would put at place `i` of its sentence, the likeliest first, then listed
    synonyms it didn't think of. All of them still face the same checks."""
    fill = _fill_model()
    masked = list(parts)
    masked[i] = fill.tokenizer.mask_token
    sentence = sentence_text(masked, bounds)
    if len(sentence.split()) > 200:  # past the model's length, where the blank could be cut off
        return []
    results = fill(sentence)
    suggested = [row["token_str"].strip() for row in results]
    return list(dict.fromkeys([*suggested, *_listed_synonyms().get(parts[i].lower(), ())]))


class MeaningGuard:
    """Whether new sentences still say what the original did: close in meaning, and entailed by it. Similarity alone
    passes a flipped sentence, since 'He won the race' and 'He lost the race' sit close together."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _similarity():
        from transformers import AutoModel, AutoTokenizer

        return AutoTokenizer.from_pretrained(SIMILARITY_MODEL), AutoModel.from_pretrained(SIMILARITY_MODEL).eval()

    def _vectors(self, sentences: list[str]):
        """Mean of the token vectors, normalized: what the model's sentence-transformers wrapper does, without
        adding that package to the environment."""
        import torch

        tokenizer, model = self._similarity()
        encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1)

    @staticmethod
    @lru_cache(maxsize=1)
    def _entailment():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL)
        return tokenizer, AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL).eval()

    def keeps(self, original: str, options: list[str], min_similarity: float = MIN_SIMILARITY,
              min_entailment: float = MIN_ENTAILMENT) -> list[bool]:
        import torch

        if not options:
            return []
        vectors = self._vectors([original, *options])
        similar = (vectors[1:] @ vectors[0]).tolist()
        tokenizer, model = self._entailment()
        encoded = tokenizer([original] * len(options), options, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            chances = model(**encoded).logits.softmax(-1)
        at = {name.lower(): i for i, name in model.config.id2label.items()}["entailment"]
        return [s >= min_similarity and e >= min_entailment
                for s, e in zip(similar, chances[:, at].tolist(), strict=True)]


@lru_cache(maxsize=1)
def _spacy():
    import spacy

    return spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])


@lru_cache(maxsize=1)
def _grammar():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL)
    return tokenizer, AutoModelForSequenceClassification.from_pretrained(GRAMMAR_MODEL).eval()


class Quality:
    """Checks a reader would make that the meaning models don't: the right part of speech and tense, grammatical
    English, and no new names or words the original gives no ground for ('slick actors', a film called a 'read')."""

    def token_at(self, sentence: str, offset: int):
        return next((t for t in _spacy()(sentence) if t.idx <= offset < t.idx + len(t)), None)

    def form(self, parts: list[str], i: int, bounds: tuple[int, int], word: str) -> str | None:
        """`word` inflected as the word at place `i`, or None if that place isn't an open-class word or `word` can't
        take its form. A name stays a name."""
        import lemminflect

        sentence = sentence_text(parts, bounds)
        token = self.token_at(sentence, len(join(parts[bounds[0]:i])))
        if token is None or token.pos_ not in CONTENT:
            return None
        lemmas = lemminflect.getLemma(word.lower(), upos=token.pos_)
        forms = lemminflect.getInflection(lemmas[0], tag=token.tag_) if lemmas else ()
        if not forms:
            return None
        trial = swapped(parts, i, forms[0])
        after = self.token_at(sentence_text(trial, bounds), len(join(trial[bounds[0]:i])))
        return forms[0] if after is not None and after.tag_ == token.tag_ else None

    def acceptability(self, sentences: list[str]) -> list[float]:
        import torch

        tokenizer, model = _grammar()
        encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return model(**encoded).logits.softmax(-1)[:, 1].tolist()

    def grammatical(self, original: str, options: list[str]) -> list[bool]:
        if not options:
            return []
        first, *rest = self.acceptability([original, *options])
        bar = min(first, MIN_ACCEPTABLE) - GRAMMAR_MARGIN
        return [value >= bar for value in rest]

    def rewrite_ok(self, original: str, option: str) -> bool:
        old, new = _spacy()(original), _spacy()(option)
        names = lambda doc: {t.text.lower() for t in doc if t.pos_ == "PROPN"}  # noqa: E731
        if names(old) != names(new):
            return False
        words = lambda doc: {t.text.lower() for t in doc if t.pos_ in CONTENT and not t.is_stop}  # noqa: E731
        known = words(old)
        return all(any(close_word(word, base) for base in known) for word in words(new) - known)


def score_all(score: Callable[[str], float], texts: list[str]) -> list[float]:
    """Several scores at once when the scorer can batch them, otherwise in turn."""
    many = getattr(score, "many", None)
    return many(texts) if many else [score(text) for text in texts]


def ranked_places(text: str, tokens: Tokens, detector, score: Callable[[str], float],
                  shortlist_size: int = SHORTLIST) -> list[int]:
    """Word places by what taking the word out really does to the score. The gradient only shortlists: on its own it
    is spread across ordinary words. The hottest windows first, since they set the document's maximum feature."""
    engine = next((c for c in getattr(detector, "classifiers", ()) if hasattr(c, "embedded_log_odds")), None)
    if engine is None:
        return []
    config = detector.config
    windows = sentence_windows(split_sentences(text), config.doc_window_words, config.doc_stride_words)
    if not windows:
        return []
    logits = engine.window_logits([text[w.start:w.end] for w in windows])
    pushes: dict[int, float] = {}
    for hot in logits.argsort()[::-1][:2].tolist():
        window = windows[hot]
        chunk = text[window.start:window.end]
        for push, start, _ in word_pushes(chunk, saliency(engine, chunk)):
            place = tokens.starts.get(window.start + start)
            if place is not None and tokens.editable(place):
                pushes[place] = max(push, pushes.get(place, 0.0))
    shortlist = sorted(pushes, key=pushes.get, reverse=True)[:shortlist_size]
    base = log_odds(score(text))
    versions = []
    for place in shortlist:
        without = list(tokens.parts)
        without[place] = ""
        versions.append(join(without))
    drops = {place: base - log_odds(value)
             for place, value in zip(shortlist, score_all(score, versions), strict=True)}
    return sorted(shortlist, key=drops.get, reverse=True)


@dataclass
class Version:
    logit: float
    parts: list[str]
    changed: int = 0


@dataclass
class ContextResult:
    text: str
    history: list[float] = field(default_factory=list)  # probabilities
    swaps: list[tuple[str, str]] = field(default_factory=list)
    passes: int = 0


def beam_search(text: str, places: list[int], score: Callable[[str], float], guard,
                proposer: Callable = propose, quality: Quality | None = None, beam: int = BEAM,
                budget: int = BUDGET, below: float = 0.1,
                max_changed: float = MAX_CHANGED, patience: int = PATIENCE) -> ContextResult:
    """Try swaps place by place, keeping the `beam` lowest-scoring versions, unchanged ones included, so a swap that
    only pays off beside a later one can survive a step. Log-odds, since near 1 the probabilities all look alike."""
    tokens = Tokens(text)
    original = tokens.parts
    first = score(text)
    spent = 1
    cap = max(1, int(max_changed * len(text.split())))
    versions = [Version(log_odds(first), original)]
    history = [first]
    limit = log_odds(below)
    stale = 0
    for place in places:
        if spent >= budget or versions[0].logit < limit or stale >= patience:
            break
        bounds = tokens.sentence_of.get(place)
        if bounds is None:
            continue
        before = sentence_text(original, bounds)
        trials: list[tuple[list[str], int]] = []
        for version in versions:
            if version.changed >= cap:
                continue
            current = sentence_text(version.parts, bounds)
            before_word, after_word = neighbours(version.parts, place)
            old = version.parts[place]
            floor = max(BIGRAM_FLOOR, MIN_PAIR_SHARE * _fit(before_word, old.lower(), after_word))
            words = [w.lower() for w in proposer(version.parts, place, bounds)
                     if allowed(old, w, current, check_form=quality is None) and close_word(old, w)
                     and _fit(before_word, w.lower(), after_word) >= floor]
            if quality:
                words = [form for w in dict.fromkeys(words) if (form := quality.form(version.parts, place, bounds, w))
                         and form.lower() != old.lower()]
            options = [swapped(version.parts, place, w) for w in words]
            sentences = [sentence_text(o, bounds) for o in options]
            kept = guard.keeps(before, sentences)
            if quality:
                kept = [a and b for a, b in zip(kept, quality.grammatical(before, sentences), strict=True)]
            passing = [o for o, ok in zip(options, kept, strict=True) if ok][:TRIED_PER_WORD]
            trials += [(option, version.changed + 1) for option in passing]
        trials = trials[:max(0, budget - spent)]
        values = score_all(score, [join(option) for option, _ in trials])
        found = [Version(log_odds(value), option, changed)
                 for (option, changed), value in zip(trials, values, strict=True)]
        spent += len(trials)
        previous = versions[0].logit
        unique = {join(v.parts): v for v in [*versions, *found]}
        versions = sorted(unique.values(), key=lambda v: v.logit)[:beam]
        if trials:  # a place with nothing to try says nothing about whether the search has stalled
            stale = 0 if versions[0].logit < previous else stale + 1
        history.append(1 / (1 + math.exp(-versions[0].logit)))
    best = versions[0]
    swaps = [(a, b) for i, (a, b) in enumerate(zip(original, best.parts, strict=True)) if i % 2 and a != b]
    return ContextResult(join(best.parts), history, swaps, spent)


def revise_above(text: str, detector, score: Callable[[str], float], above: float, name: str = "t5") -> ContextResult:
    """No API: rewrite only the sentences the detector scores at `above` or more, on this computer, under the meaning
    and grammar checks. Weaker than a hosted model, and it keeps whatever the checks refuse as it was."""
    report = detector.analyze(text)
    targets = {report.text[s.start:s.end].strip() for s in report.sentences if s.ai_probability >= above}
    return paraphrase_stage(report.text, score, MeaningGuard(), quality=Quality(), name=name, targets=targets)


def contextual_attack(text: str, detector, score: Callable[[str], float], guard=None, budget: int = BUDGET,
                      below: float = 0.1, beam: int = BEAM, patience: int = PATIENCE,
                      shortlist: int = SHORTLIST) -> ContextResult:
    """The whole stage on one text: measure which words matter, then search their swaps."""
    clean = normalize(text)[0]
    guard = guard or MeaningGuard()
    places = ranked_places(clean, Tokens(clean), detector, score, shortlist)
    return beam_search(clean, places, score, guard, quality=Quality(), beam=beam, budget=budget, below=below,
                       patience=patience)


@lru_cache(maxsize=1)
def _paraphraser(name: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(DIPPER_TOKENIZER if name == "dipper" else PARAPHRASERS[name])
    return tokenizer, AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASERS[name]).to(device).eval(), device


def paraphrase_prompt(name: str, sentence: str, context: str = "", lexical: int = 40, order: int = 40) -> str:
    """DIPPER takes how much to change, in steps of 20, and may see the text before the sentence."""
    if name == "dipper":
        return f"lexical = {lexical}, order = {order} {context} <sent> {sentence} </sent>".replace("  ", " ")
    return sentence if name == "pegasus" else f"paraphrase: {sentence}"


def paraphrase(name: str, sentences: list[str], contexts: list[str] | None = None, options: int = 3) -> list[list[str]]:
    """A few rewrites of each sentence."""
    import torch

    tokenizer, model, device = _paraphraser(name)
    prompts = [paraphrase_prompt(name, s, (contexts or [""] * len(sentences))[i]) for i, s in enumerate(sentences)]
    # Pegasus reads 60 tokens at most. A longer sentence is left alone: rewriting its first part would lose the rest.
    limit = getattr(model.config, "max_position_embeddings", 512) or 512
    if any(len(ids) > limit for ids in tokenizer(prompts)["input_ids"]):
        return [[] for _ in sentences]
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=min(limit, 512),
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=True, top_p=0.75, num_return_sequences=options, max_new_tokens=128)
    texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    return [[t.strip() for t in texts[i * options:(i + 1) * options]] for i in range(len(sentences))]


def paraphrase_stage(text: str, score: Callable[[str], float], guard, rewrite: Callable | None = None,
                     quality: Quality | None = None,
                     name: str = "t5", budget: int = BUDGET, below: float = 0.1,
                     max_sentences: int | None = None, targets: set[str] | None = None) -> ContextResult:
    """Rewrite one sentence at a time, keeping a rewrite only if the meaning checks pass, its numbers and negations
    stay, and the whole text then scores lower."""
    rewrite = rewrite or (lambda sentences, contexts: paraphrase(name, sentences, contexts))
    engine = _engine()
    current = text
    best = score(current)
    history, spent = [best], 1
    at = tried = 0
    while spent < budget and best >= below and (max_sentences is None or tried < max_sentences):
        spans = [s for s in split_sentences(current) if s.start >= at]
        if not spans:
            break
        start, end = spans[0].start, spans[0].end
        at = end
        sentence = current[start:end]
        if len(sentence.split()) < 6 or (targets is not None and sentence.strip() not in targets):
            continue
        tried += 1
        options = rewrite([sentence], [current[max(0, start - 300):start]])[0]
        facts = engine._invariants(sentence)
        options = [o for o in dict.fromkeys(options) if o and o != sentence
                   and engine._invariants(o)[0] == facts[0] and len(engine._invariants(o)[1]) == len(facts[1])]
        if quality:
            options = [o for o in options if quality.rewrite_ok(sentence, o)]
        kept = guard.keeps(sentence, options, min_similarity=PARAPHRASE_SIMILARITY,
                           min_entailment=PARAPHRASE_ENTAILMENT)
        if quality:
            kept = [a and b for a, b in zip(kept, quality.grammatical(sentence, options), strict=True)]
        trials = [(current[:start] + o + current[end:], o) for o, ok in zip(options, kept, strict=True) if ok]
        trials = trials[:max(0, budget - spent)]
        values = score_all(score, [trial for trial, _ in trials])
        spent += len(trials)
        if values and min(values) < best:
            best = min(values)
            current, option = trials[values.index(best)]
            at = start + len(option)
            history.append(best)
    return ContextResult(current, history, passes=spent)


def guard_check(count: int = 400) -> None:
    """How MeaningGuard does on PAWS: pairs that share nearly all their words, half of them not paraphrases, which is
    the case that fools a similarity score. Prints each bar's share of true paraphrases passed and of false ones."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download("google-research-datasets/paws", repo_type="dataset", local_files_only=True))
    frame = pd.read_parquet(next(root.glob("labeled_final/test*.parquet"))).head(count)
    guard = MeaningGuard()
    similar, entailed = [], []
    for row in frame.itertuples():
        vectors = guard._vectors([row.sentence1, row.sentence2])
        similar.append(float(vectors[0] @ vectors[1]))
        guard_row = guard._entailment()
        import torch

        encoded = guard_row[0]([row.sentence1], [row.sentence2], truncation=True, return_tensors="pt")
        with torch.no_grad():
            chances = guard_row[1](**encoded).logits.softmax(-1)[0]
        at = {name.lower(): i for i, name in guard_row[1].config.id2label.items()}["entailment"]
        entailed.append(float(chances[at]))
    frame = frame.assign(similar=similar, entailed=entailed)
    for s_bar, e_bar in ((0.75, 0.5), (0.85, 0.5), (0.85, 0.7), (0.9, 0.7), (0.9, 0.9), (0.95, 0.9)):
        passed = (frame.similar >= s_bar) & (frame.entailed >= e_bar)
        print(f"similarity {s_bar}, entailment {e_bar}: true paraphrases passed {passed[frame.label == 1].mean():.0%}, "
              f"false ones passed {passed[frame.label == 0].mean():.0%}")


if __name__ == "__main__":
    guard_check()
