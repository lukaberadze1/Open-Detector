"""Keep a living style profile of the writing in an Obsidian vault, stacked week on week.

    python update_style_profile.py            # scan the vault, measure, snapshot the profile
    python update_style_profile.py --check    # the profile still has the template's exact format

Every paragraph the writer has written is kept in data/styles/alexandria/corpus.jsonl with the ISO
week it first appeared (on the first run, the week its note was created). Nothing is dropped: a
paragraph deleted from the vault stays in the corpus, and only a paragraph edited in place gives
way to its new version. The measurements weight each week by its age (half-life HALF_LIFE weeks,
never below FLOOR), and the profile the writer started from counts as one more batch of
BASELINE["words"] words dated to the first run. So recent writing leads and older writing keeps a say.

Only the writer's own words are measured. Left out: frontmatter, headings, callouts and quotes
(where the vault keeps review notes and pasted sources), embeds, tables, code, HTML comments,
==highlighted== passages, task boxes, template stubs, Russian text, templates, and the vault's
top-level "00 " guide and review-log notes.

The script writes the numbers. The prose of data/styles/alexandria_profile.json is revised by
Claude from them each week, following .claude/skills/style-profile/SKILL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from opendetector.segment import split_sentences

VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/The Alexandria Archive"
STATE = Path("data/styles/alexandria")
PROFILE = Path("data/styles/alexandria_profile.json")
TEMPLATE = Path("data/styles/pasted_style_profile.json")
DICTIONARY = Path("/usr/share/dict/words")

HALF_LIFE = 12  # weeks until a week's writing counts half
FLOOR = 0.25  # the least any week ever counts
# The measured profile the writer started from: about 54,000 words of essays, fiction, reports and notes.
BASELINE = {"words": 54000, "mean": 18.3, "min": 2, "max": 106,
            "pct": {"under_6": 6.9, "6_to_15": 39.3, "16_to_29": 40.9, "30_plus": 13.0}}
EDIT_SIMILARITY = 0.8  # a new paragraph this close to a vanished one in the same note is an edit of it
MIN_WORDS, MIN_LIST_WORDS = 4, 6
SAMPLE_WORDS = 8000  # how much of the new writing goes into new_this_week.md
SKIP_DIRS = {"80 Scribe Tools", "99 The Reliquary"}

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


# -- Reading the vault -----------------------------------------------------------------

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
CREATED = re.compile(r"^created:\s*[\"']?(\d{1,4})[-./](\d{1,2})[-./](\d{1,4})", re.M)
COMMENT = re.compile(r"<!--.*?-->", re.S)
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
EMBED = re.compile(r"!\[\[[^\]]*\]\]|!\[[^\]]*\]\([^)]*\)")
WIKILINK = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
MDLINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
CODE = re.compile(r"`[^`\n]*`")
# The writer types straight quotes, no dashes and no bullet characters, and marks what they paste with
# a backtick at the start of the line. So a line carrying any of these was pasted from a book, a page
# or a chatbot, or is the vault's own link list ("People — A · B").
PASTED = re.compile(r"^`|[“”‘’•·→—–`]")
BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
ITALIC = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])")
FOOTNOTE = re.compile(r"\[\^[^\]]*\]:?\s*")
BLOCK_ID = re.compile(r"\s\^[\w-]+$")
LIST = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
TASK = re.compile(r"^[-*+]\s+\[.\]")
# A rule, a template's [Name] placeholder, or a line in italics throughout (a caption, or a guide's prompt).
SKIP_LINE = re.compile(r"([-*_]\s*){3,}|\[[^\]]*\]|\*[^*].*\*|_[^_].*_")
NON_LATIN = re.compile(r"[^\W\d_a-zA-ZÀ-ɏ]")  # Russian and Georgian letters, for one


def iso_week(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def week_weight(week: str, now: str) -> float:
    def number(w: str) -> int:
        year, n = w.split("-W")
        return date.fromisocalendar(int(year), int(n), 1).toordinal() // 7

    age = max(0, number(now) - number(week))
    return max(0.5 ** (age / HALF_LIFE), FLOOR)


def _created_week(frontmatter: str) -> str | None:
    match = CREATED.search(frontmatter)
    if not match:
        return None
    a, b, c = match.groups()
    year, month, day = (a, b, c) if len(a) == 4 else (c, b, a)
    try:
        return iso_week(date(int(year), int(month), int(day)))
    except ValueError:
        return None


def own_paragraphs(body: str) -> tuple[list[str], list[str]]:
    """The lines of a note body the writer wrote, cleaned of Markdown, and the lines skipped as
    pasted. In Obsidian every line is its own paragraph, and the writer types one thought per line."""
    paragraphs, pasted = [], []
    quoted = False
    for line in FENCE.sub("", COMMENT.sub("", body)).split("\n"):
        line = line.strip()
        # A quote or callout runs on, without its ">", until the next blank line.
        quoted = bool(line) and (quoted or line.startswith(">"))
        if (quoted or not line or line.startswith(("#", "|")) or "==" in line or "🧭" in line
                or TASK.match(line) or SKIP_LINE.fullmatch(line)):
            continue
        listed = bool(LIST.match(line))
        if PASTED.match(line):
            pasted.append(line)
            continue
        line = CODE.sub("", EMBED.sub("", LIST.sub("", line)))
        if PASTED.search(WIKILINK.sub("", line)):  # link targets may contain dashes; the writer's text may not
            pasted.append(line)
            continue
        line = WIKILINK.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(1).split("#")[0], line)
        line = BLOCK_ID.sub("", FOOTNOTE.sub("", MDLINK.sub(r"\1", line)).rstrip())
        line = ITALIC.sub(r"\1", BOLD.sub(r"\2", line)).strip()
        if line.endswith(":") or len(WORD_RE.findall(line)) < (MIN_LIST_WORDS if listed else MIN_WORDS):
            continue  # an empty template label, or too short to say anything about style
        if len(NON_LATIN.findall(line)) > 0.2 * sum(ch.isalpha() for ch in line):
            continue
        paragraphs.append(line)
    return paragraphs, pasted


def parse_note(markdown: str) -> tuple[str | None, list[str], list[str]]:
    """The week the note was created (None without a `created:` property), its own paragraphs, and
    the lines skipped as pasted."""
    match = FRONTMATTER.match(markdown)
    if not match:
        return None, *own_paragraphs(markdown)
    return _created_week(match.group(1)), *own_paragraphs(markdown[match.end():])


def scan(vault: Path) -> tuple[list[tuple[str, str, list[str]]], list[str], list[str]]:
    """(note, week written, paragraphs) for each note, the notes that could not be read, and the
    lines skipped as pasted."""
    notes, unreadable, pasted = [], [], []
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if len(rel.parts) == 1 and rel.name.startswith("00 "):
            continue  # the vault's guide and review logs
        try:
            created, paragraphs, skipped = parse_note(path.read_text(encoding="utf-8", errors="replace"))
            written = created or iso_week(date.fromtimestamp(path.stat().st_mtime))
        except OSError:  # an iCloud file not downloaded yet
            unreadable.append(rel.as_posix())
            continue
        notes.append((rel.as_posix(), written, paragraphs))
        pasted += skipped
    return notes, unreadable, pasted


# -- The corpus ------------------------------------------------------------------------

def paragraph_id(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_corpus(corpus: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in corpus), encoding="utf-8")
    os.replace(tmp, path)  # the corpus is the only memory, so never leave it half written


def update_corpus(corpus: list[dict], notes: list[tuple[str, str, list[str]]], week: str) -> dict:
    """Add every paragraph not seen before, dated `week` (on the first run, by when its note was
    written). A paragraph gone from the vault is kept, unless a new paragraph in the same note is
    an edit of it; then the old version is marked superseded and no longer measured."""
    seed = not corpus
    known = {e["id"] for e in corpus}
    present, added = set(), []
    for note, written, paragraphs in notes:
        for text in paragraphs:
            pid = paragraph_id(text)
            present.add(pid)
            if pid in known:
                continue
            known.add(pid)
            entry = {"id": pid, "week": written if seed else week, "note": note,
                     "words": len(WORD_RE.findall(text)), "text": text}
            corpus.append(entry)
            added.append(entry)

    vanished = defaultdict(list)
    for e in corpus:
        if e["id"] not in present and "superseded" not in e:
            vanished[e["note"]].append(e)
    edited = 0
    for entry in added:
        for old in vanished.get(entry["note"], []):
            if "superseded" not in old and SequenceMatcher(None, old["text"], entry["text"]).ratio() >= EDIT_SIMILARITY:
                old["superseded"] = entry["week"]
                entry["edit_of"] = old["id"]
                edited += 1
                break
    kept = sum(e["id"] not in present and "superseded" not in e for e in corpus)
    return {"seed": seed, "added": added, "edited": edited, "kept_after_deletion": kept}


# -- Measuring -------------------------------------------------------------------------

def _features(*specs: tuple[str, str, bool]) -> dict[str, re.Pattern]:
    return {name: re.compile(pattern, 0 if exact else re.IGNORECASE) for name, pattern, exact in specs}


_INITIAL = r"(?:^|[.!?]\s+)"
_VOWEL = r"(?!uni|use|usu|uti|eu|one|once|ur)[aeiou]"  # a vowel letter with a vowel sound
FEATURES = _features(
    ("em_dash", "—", True), ("en_dash", "–", True), ("parenthesis", r"\(", True),
    ("semicolon", ";", True), ("colon", r":(?!//)", True), ("question_mark", r"\?", True),
    ("exclamation_mark", "!", True), ("space_before_punctuation", r"\w [.,;:!?](?=\s|$)", True),
    ("double_space", r"\S  +\S", True), ("slash_pair", r"[^\W\d_]+/[^\W\d_]+", True),
    ("however_initial", _INITIAL + r"However\b", True), ("however_comma", _INITIAL + r"However,", True),
    ("yet_initial", _INITIAL + r"Yet\b", True), ("yet_comma", _INITIAL + r"Yet,", True),
    ("a_before_vowel", r"\ba " + _VOWEL, False), ("an_before_vowel", r"\ban " + _VOWEL, False),
    ("whether", r"\bwhether\b", False), ("wether", r"\bwether\b", False),
    ("its_for_it_is", r"\bits (?:a|an|the|not|how|what|easier|clear|safe|no|like|very|so|also|just|been|going|"
                      r"similar|because|hard|important|true|why|when|all|more|only)\b", False),
    ("it_s", r"\bit['’]s\b", False),
    ("contraction_without_apostrophe", r"\b(?:dont|doesnt|didnt|cant|wont|isnt|wasnt|arent|werent|thats|whats|"
                                       r"shes|theyre|youre|ive|im|couldnt|wouldnt|shouldnt|hasnt|havent)\b", False),
    ("contraction_with_apostrophe", r"\b(?:(?:do|does|did|ca|wo|is|was|are|were|could|would|should|has|have)n['’]t|"
                                    r"(?:that|what|she|it|let)['’]s|(?:they|you|we)['’]re|i['’](?:m|ve))\b", False),
    ("then_for_than", r"\b(?:more|less|better|worse|rather|larger|smaller|greater|bigger|faster|higher|lower)\b"
                      r"(?:\s+\w+){0,3}\s+then\b", False),
    ("lowercase_i", r"(?<![\w'’])i(?![\w'’])", True),
    ("clearly", r"\bclearly\b", False), ("evident", r"\bevident\b", False),
    ("this_shows", r"\b(?:this|it) shows\b", False), ("this_is_why", r"\bthis is why\b", False),
    ("we_can", r"\bwe can\b", False), ("simply", r"\bsimply\b", False), ("really", r"\breally\b", False),
    ("in_order_to", r"\bin order to\b", False), ("a_lot_of", r"\ba lot of\b", False),
    ("no_matter", r"\bno matter\b", False), ("even", r"\beven\b", False), ("obviously", r"\bobviously\b", False),
    ("i_believe", r"\bi believe\b", False), ("referred_to_as", r"\bref+er+ed to as\b", False),
    ("which", r"\bwhich\b", False), ("and", r"\band\b", False), ("because", r"\bbecause\b", False),
    ("pronoun_I", r"\bI\b", True), ("we_us_our", r"\b(?:we|us|our|ours)\b", False),
    ("you_your", r"\b(?:you|your|yours)\b", False), ("he_him_his", r"\b(?:he|him|his)\b", False),
    ("they_them_their", r"\b(?:they|them|their)\b", False),
    ("be_verbs", r"\b(?:am|is|are|was|were|be|been|being)\b", False),
    ("modals", r"\b(?:can|could|will|would|shall|should|may|might|must)\b", False),
)
_SUFFIXES = (("ies", "y"), ("ied", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""),
             ("ing", "e"), ("ly", ""), ("er", ""), ("er", "e"), ("est", ""), ("'s", ""), ("’s", ""),
             ("isation", "ization"), ("ise", "ize"), ("ised", "ized"), ("ising", "izing"), ("tre", "ter"),
             ("our", "or"), ("ours", "ors"))
# Common words the system dictionary (Webster's 2nd) has no entry for.
_EXTRA_WORDS = frozenset(["held", "began", "became", "women", "withdrew", "feet", "forgave", "woken", "etc", "poleis"])


@lru_cache(maxsize=1)
def dictionary() -> frozenset[str]:
    if not DICTIONARY.exists():
        return frozenset()
    return _EXTRA_WORDS | {w.lower() for w in DICTIONARY.read_text(encoding="utf-8", errors="ignore").split()}


def known_word(word: str, words: frozenset[str]) -> bool:
    if word in words:
        return True
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix):
            base = word[: -len(suffix)] + repl
            if base in words or (len(base) > 3 and base[-1] == base[-2] and base[:-1] in words):
                return True
    return False


def banned_patterns(profile: dict) -> dict[str, re.Pattern]:
    """A pattern per literal phrase in the profile's banned list (not 'X' templates or glossed entries)."""
    phrases = (profile.get("style_profile", profile).get("vocabulary_and_lexical_density", {})
               .get("banned_ai_cliches", []))
    return {p: re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in phrases
            if "(" not in p and not re.search(r"\b[XY]\b", p)}


def mattr(tokens: list[str], window: int = 100) -> float | None:
    if len(tokens) < window:
        return None
    seen = Counter(tokens[:window])
    total = len(seen)
    for i in range(window, len(tokens)):
        out = tokens[i - window]
        seen[out] -= 1
        if not seen[out]:
            del seen[out]
        seen[tokens[i]] += 1
        total += len(seen)
    return total / (len(tokens) - window + 1) / window


def measure(texts: list[str], banned: dict[str, re.Pattern]) -> dict:
    """Additive counts for a batch of paragraphs, so batches can be weighted and summed."""
    lengths, openers, features, banned_hits, unknown = Counter(), Counter(), Counter(), Counter(), Counter()
    pairs = pair_diff = pair_jumps = commas = 0
    for text in texts:
        sizes = []
        for s in split_sentences(text):
            tokens = WORD_RE.findall(text[s.start : s.end])
            if tokens:
                sizes.append(len(tokens))
                openers[tokens[0] if tokens[0] == "I" else tokens[0].lower()] += 1
        lengths.update(sizes)
        for a, b in pairwise(sizes):
            pairs += 1
            pair_diff += abs(a - b)
            pair_jumps += max(a, b) >= 2.5 * min(a, b)
        commas += text.count(",")
        for name, rx in FEATURES.items():
            features[name] += len(rx.findall(text))
        for phrase, rx in banned.items():
            banned_hits[phrase] += len(rx.findall(text))

    tokens = WORD_RE.findall("\n".join(texts))
    lowered = [t.lower() for t in tokens]
    types = Counter(lowered)
    words = dictionary()
    checked = [t for t in tokens if t.islower() and len(t) >= 3] if words else []
    unknown.update(t for t in checked if not known_word(t.replace("’", "'"), words))
    value = mattr(lowered)
    return {
        "paragraphs": len(texts), "words": len(tokens), "letters": sum(map(len, tokens)),
        "long_words": sum(len(t) >= 7 for t in tokens), "sentences": sum(lengths.values()), "lengths": lengths,
        "pairs": pairs, "pair_diff": pair_diff, "pair_jumps": pair_jumps, "commas": commas,
        "openers": openers, "features": features, "banned": +banned_hits,
        "mattr_sum": (value or 0) * len(tokens), "mattr_words": len(tokens) if value is not None else 0,
        "hapax_sum": (sum(c == 1 for c in types.values()) / len(types) * len(tokens)) if types else 0,
        "checked_words": len(checked), "unknown": unknown,
    }


def combine(parts: list[tuple[dict, float]]) -> dict:
    """Weighted sum of measure() results."""
    total: dict = {}
    for counts, weight in parts:
        for key, value in counts.items():
            if isinstance(value, Counter):
                bucket = total.setdefault(key, Counter())
                for k, v in value.items():
                    bucket[k] += v * weight
            else:
                total[key] = total.get(key, 0) + value * weight
    return total


def _num(x: float) -> float | int:
    return round(x) if abs(x - round(x)) < 0.05 else round(x, 1)


def _pct(part: float, whole: float) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def summarize(c: dict) -> dict | None:
    """Rates, shares and distributions from (possibly weighted) counts."""
    if not c or not c.get("sentences"):
        return None
    n, words, f = c["sentences"], c["words"], c["features"]
    lengths = sorted((k, v) for k, v in c["lengths"].items() if v > 0)
    mean = sum(k * v for k, v in lengths) / n
    std = math.sqrt(sum(v * (k - mean) ** 2 for k, v in lengths) / n)
    running, median = 0.0, lengths[-1][0]
    for k, v in lengths:
        running += v
        if running >= n / 2:
            median = k
            break

    def share(lo: int, hi: int | None = None) -> float:
        return _pct(sum(v for k, v in lengths if k >= lo and (hi is None or k <= hi)), n)

    a, an = f["a_before_vowel"], f["an_before_vowel"]
    bare, marked = f["contraction_without_apostrophe"], f["contraction_with_apostrophe"]
    return {
        "words": round(words), "sentences": round(n), "paragraphs": round(c["paragraphs"]),
        "sentence_length": {
            "mean": round(mean, 1), "median": median, "std": round(std, 1),
            "coefficient_of_variation": round(std / mean, 2), "min": lengths[0][0], "max": lengths[-1][0],
            "adjacent_mean_difference": round(c["pair_diff"] / c["pairs"], 1) if c["pairs"] else None,
            "adjacent_pairs_2_5x_pct": _pct(c["pair_jumps"], c["pairs"]),
            "distribution_pct": {"under_6": share(1, 5), "6_to_15": share(6, 15), "16_to_29": share(16, 29),
                                 "30_plus": share(30), "45_plus": share(45)},
        },
        "commas_per_sentence": round(c["commas"] / n, 2),
        "vocabulary": {
            "mattr_100": round(c["mattr_sum"] / c["mattr_words"], 3) if c["mattr_words"] else None,
            "hapax_pct_of_types": _pct(c["hapax_sum"], words),
            "mean_word_length": round(c["letters"] / words, 2),
            "words_7_plus_letters_pct": _pct(c["long_words"], words),
        },
        "per_1000_words": {name: round(v * 1000 / words, 2) for name, v in f.items()},
        "sentence_openers_pct": {w: _pct(v, n) for w, v in c["openers"].most_common(15)},
        "mechanics": {
            "however_followed_by_comma": f"{_num(f['however_comma'])} of {_num(f['however_initial'])}",
            "yet_followed_by_comma": f"{_num(f['yet_comma'])} of {_num(f['yet_initial'])}",
            "a_before_vowel_sound_pct": _pct(a, a + an),
            "a_vs_an": f"{_num(a)} a, {_num(an)} an",
            "wether_vs_whether": f"{_num(f['wether'])} wether, {_num(f['whether'])} whether",
            "its_for_it_is_vs_it_s": f"{_num(f['its_for_it_is'])} its, {_num(f['it_s'])} it's",
            "contractions_without_apostrophe_pct": _pct(bare, bare + marked),
        },
        "spelling": {
            "unknown_lowercase_word_pct": _pct(sum(c["unknown"].values()), c["checked_words"]),
            "note": "words not in the system dictionary after stripping common endings; includes real slips "
                    "and rare or modern words, so read the list rather than trusting the rate",
            "most_common_unknown": [[w, _num(v)] for w, v in c["unknown"].most_common(40)],
        },
        "banned_phrases_used": {p: _num(v) for p, v in c["banned"].most_common() if v > 0},
    }


def profile_numbers(stacked: dict, baseline_weight: float) -> dict:
    """The profile's numeric fields: the stacked archive blended with the starting profile, each
    sentence counting once and the starting profile's sentences weighted by its age."""
    s = summarize(stacked)
    if s is None:
        base, pct = BASELINE, BASELINE["pct"]
        return {"avg_sentence_length_words": base["mean"], "min_max_sentence_length": [base["min"], base["max"]],
                "sentence_length_distribution": _distribution(pct), "starting_profile_share_pct": 100.0}
    nb = BASELINE["words"] / BASELINE["mean"] * baseline_weight
    na = stacked["sentences"]
    sl = s["sentence_length"]

    def blend(base: float, archive: float) -> float:
        return round((base * nb + archive * na) / (nb + na), 1)

    pct = {k: blend(v, sl["distribution_pct"][k]) for k, v in BASELINE["pct"].items()}
    return {
        "avg_sentence_length_words": blend(BASELINE["mean"], sl["mean"]),
        "min_max_sentence_length": [min(BASELINE["min"], sl["min"]), max(BASELINE["max"], sl["max"])],
        "sentence_length_distribution": _distribution(pct),
        "starting_profile_share_pct": _pct(nb, nb + na),
    }


def _distribution(pct: dict) -> dict:
    return {"under_6_words_pct": f"{pct['under_6']}%", "6_to_15_words_pct": f"{pct['6_to_15']}%",
            "16_plus_words_pct": f"{round(pct['16_to_29'] + pct['30_plus'], 1)}% (16-29 words: {pct['16_to_29']}%, "
                                 f"30+ words: {pct['30_plus']}%)"}


DRIFT = [("mean sentence length", ("sentence_length", "mean")),
         ("coefficient of variation", ("sentence_length", "coefficient_of_variation")),
         ("sentences under 6 words %", ("sentence_length", "distribution_pct", "under_6")),
         ("sentences 30+ words %", ("sentence_length", "distribution_pct", "30_plus")),
         ("commas per sentence", ("commas_per_sentence",)), ("MATTR-100", ("vocabulary", "mattr_100")),
         ("unknown words %", ("spelling", "unknown_lowercase_word_pct")),
         ("'a' before vowel sounds %", ("mechanics", "a_before_vowel_sound_pct")),
         ("contractions without apostrophe %", ("mechanics", "contractions_without_apostrophe_pct")),
         ("em-dashes per 1000", ("per_1000_words", "em_dash")),
         ("parentheses per 1000", ("per_1000_words", "parenthesis")),
         ("'which' per 1000", ("per_1000_words", "which"))]


def measurements(corpus: list[dict], change: dict, week: str, baseline_week: str,
                 banned: dict[str, re.Pattern]) -> dict:
    by_week = defaultdict(list)
    for e in corpus:
        if "superseded" not in e:
            by_week[e["week"]].append(e["text"])
    measured = {w: measure(texts, banned) for w, texts in sorted(by_week.items())}
    weights = {w: week_weight(w, week) for w in measured}
    stacked = combine([(c, weights[w]) for w, c in measured.items()])
    new = [] if change["seed"] else [e["text"] for e in change["added"]]
    this_week = summarize(measure(new, banned)) if new else None
    stacked_summary = summarize(stacked)

    drift = {}
    if this_week and stacked_summary and this_week["words"] >= 150:
        for label, path in DRIFT:
            values = []
            for summary in (stacked_summary, this_week):
                for key in path:
                    summary = summary.get(key) if isinstance(summary, dict) else None
                values.append(summary)
            drift[label] = {"stacked": values[0], "this_week": values[1]}

    weeks = [{"week": "starting profile", "words": BASELINE["words"],
              "weight": round(week_weight(baseline_week, week), 2), "mean_sentence_length": BASELINE["mean"]}]
    for w, c in measured.items():
        s = summarize(c)
        weeks.append({"week": w, "words": c["words"], "weight": round(weights[w], 2),
                      "mean_sentence_length": s and s["sentence_length"]["mean"],
                      "coefficient_of_variation": s and s["sentence_length"]["coefficient_of_variation"]})
    live = [e for e in corpus if "superseded" not in e]
    return {
        "updated": date.today().isoformat(), "week": week,
        "settings": {"half_life_weeks": HALF_LIFE, "floor": FLOOR, "starting_profile_words": BASELINE["words"],
                     "starting_profile_week": baseline_week},
        "corpus": {"paragraphs": len(live), "words": sum(e["words"] for e in live), "weeks": len(measured),
                   "first_run": change["seed"], "new_paragraphs": len(change["added"]),
                   "new_words": sum(e["words"] for e in change["added"]), "edited_paragraphs": change["edited"],
                   "kept_after_deletion_from_vault": change["kept_after_deletion"]},
        "profile_numbers": profile_numbers(stacked, week_weight(baseline_week, week)),
        "stacked": stacked_summary,
        "this_week": this_week,
        "drift": drift,
        "all_time_unweighted": summarize(combine([(c, 1) for c in measured.values()])),
        "weeks": weeks,
    }


# -- Outputs ---------------------------------------------------------------------------

def write_new_writing(path: Path, change: dict, week: str) -> None:
    added = sorted(change["added"], key=lambda e: e["week"], reverse=True)
    lines = [f"# New writing, {week}", ""]
    if change["seed"]:
        lines += ["First run: the whole archive was added, each paragraph dated by when its note was written. "
                  f"Below are the most recent paragraphs, up to {SAMPLE_WORDS} words.", ""]
    elif not added:
        lines += ["No new paragraphs since the last run.", ""]
    total, by_note = 0, defaultdict(list)
    for e in added:
        if total + e["words"] > SAMPLE_WORDS and total:
            break
        total += e["words"]
        by_note[e["note"]].append(e)
    if added and not change["seed"]:
        lines += [f"{len(added)} paragraphs, {sum(e['words'] for e in added)} words. Edits of older paragraphs "
                  "are marked (edited).", ""]
    for note, entries in by_note.items():
        lines += [f"## {note}", ""]
        for e in entries:
            lines += [("(edited) " if "edit_of" in e else "") + e["text"], ""]
    if len(added) > sum(map(len, by_note.values())):
        lines.append(f"({len(added) - sum(map(len, by_note.values()))} more paragraphs not shown.)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_profile(profile: Path, template: Path, history: Path, week: str) -> Path | None:
    """Start the living profile from the template once, and keep a copy of it as it stood at the
    start of each week, so every week's edit can be compared or undone."""
    if not profile.exists():
        profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, profile)
    snapshot = history / f"{profile.stem}_{week}.json"
    if snapshot.exists():
        return None
    history.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(profile, snapshot)
    return snapshot


def _kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return {dict: "object", list: "list", str: "string"}.get(type(value), type(value).__name__)


def format_errors(profile, template, path: str = "") -> list[str]:
    """Where the profile departs from the template's format: its keys, their order, value types."""
    where = path or "top level"
    if _kind(profile) != _kind(template):
        return [f"{where}: expected {_kind(template)}, found {_kind(profile)}"]
    if isinstance(template, dict):
        errors = []
        missing = [k for k in template if k not in profile]
        extra = [k for k in profile if k not in template]
        if missing or extra:
            errors.append(f"{where}: missing keys {missing}, unexpected keys {extra}")
        elif list(profile) != list(template):
            errors.append(f"{where}: keys are out of order")
        for key in template:
            if key in profile:
                errors += format_errors(profile[key], template[key], f"{path}.{key}" if path else key)
        return errors
    if isinstance(template, list):
        kinds = {_kind(x) for x in template}
        if kinds == {"number"} and len(profile) != len(template):
            return [f"{where}: expected {len(template)} numbers, found {len(profile)}"]
        return [f"{where}: unexpected {_kind(x)} item {str(x)[:40]!r}" for x in profile if _kind(x) not in kinds][:3]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vault", type=Path, default=Path(os.environ.get("ALEXANDRIA_VAULT", VAULT)))
    parser.add_argument("--state", type=Path, default=STATE, help="corpus, measurements and history")
    parser.add_argument("--profile", type=Path, default=PROFILE, help="the living style profile")
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="the profile whose format is kept")
    parser.add_argument("--week", help="ISO week to file new writing under, e.g. 2026-W44 (default: this week)")
    parser.add_argument("--check", action="store_true", help="only check the profile's format")
    args = parser.parse_args(argv)

    if args.check:
        try:
            errors = format_errors(json.loads(args.profile.read_text()), json.loads(args.template.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            errors = [f"could not read the profile: {e}"]
        for error in errors:
            print(error, file=sys.stderr)
        print("format OK" if not errors else f"{len(errors)} format problem(s)")
        return 1 if errors else 0

    if not args.vault.is_dir():
        parser.error(f"vault not found: {args.vault}")
    if not args.profile.exists() and not args.template.exists():
        parser.error(f"neither the profile {args.profile} nor the template {args.template} exists")
    week = args.week or iso_week(date.today())
    corpus_path = args.state / "corpus.jsonl"
    corpus = load_corpus(corpus_path)
    notes, unreadable, pasted = scan(args.vault)
    change = update_corpus(corpus, notes, week)
    save_corpus(corpus, corpus_path)

    meta_path = args.state / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"starting_profile_week": week}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    snapshot = prepare_profile(args.profile, args.template, args.state / "history", week)
    banned = banned_patterns(json.loads(args.profile.read_text()))
    report = measurements(corpus, change, week, meta["starting_profile_week"], banned)
    report["unreadable_notes"] = unreadable
    report["skipped_as_pasted"] = {"lines": len(pasted), "examples": [line[:120] for line in pasted[:8]]}
    (args.state / "measurements.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_new_writing(args.state / "new_this_week.md", change, week)
    changelog = args.state / "CHANGELOG.md"
    if not changelog.exists():
        changelog.write_text("# Style profile changelog\n\nNewest week at the bottom.\n")

    c, numbers = report["corpus"], report["profile_numbers"]
    print(f"{len(notes)} notes read{f', {len(unreadable)} unreadable' if unreadable else ''}. "
          f"Corpus: {c['paragraphs']} paragraphs, {c['words']} words over {c['weeks']} weeks.")
    print(("First run: " if c["first_run"] else f"{week}: ")
          + f"{c['new_paragraphs']} new paragraphs ({c['new_words']} words), {c['edited_paragraphs']} edited, "
            f"{c['kept_after_deletion_from_vault']} kept after deletion from the vault.")
    print(f"Profile numbers: mean sentence {numbers['avg_sentence_length_words']} words, "
          f"range {numbers['min_max_sentence_length']}, starting profile {numbers['starting_profile_share_pct']}% "
          "of the blend.")
    print(f"Wrote {args.state / 'measurements.json'} and {args.state / 'new_this_week.md'}"
          + (f"; snapshot {snapshot}" if snapshot else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
