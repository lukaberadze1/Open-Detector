"""Measure red-team generation settings on a document, four measures each, for the Red team page.

    python -m eval.tune_generation report.docx                 # measure, then write the table
    python -m eval.tune_generation report.docx --models gemini-3.5-flash-lite --repeats 1
    python -m eval.tune_generation --fit-only                  # rewrite the table from the trials so far

A setting is a model, its thinking level and how many blocks go in one request. Each trial rewrites
the same run of the document's blocks (at least --words words) the way a Red team batch does, with the
same prompts and the style profile as the system prompt, and records:

- detector: the AI likelihood of the rewritten run, from the --profile build of the detector
- style: how closely the rewrite follows the style profile (see StyleYardstick)
- speed: seconds of writing per block, counting only the request that answered: a 503 says how busy
  the server was that minute, not how fast the setting is
- quota: requests per block, Gemini's own count of input, output and thinking tokens per block, and
  each model's daily allowance of requests: Gemini's quota errors state it ("limit: 20 requests per
  day"), so a model that runs out records its limit in eval/cache/tuning_limits.json, and
  --daily-limit MODEL=N sets one by hand (AI Studio lists a project's limits)

The run holds at least two requests' worth of the largest group, so a group size is measured as a
batch sends it, not as one request holding a short document whole.

Measuring every combination costs more requests than the free tier allows, so by default (--design
main) each model is measured at each thinking level with the largest group, and the group sizes with
one model (--baseline-model) at low thinking. The other combinations are predicted on the assumption
that the effects add up: a group size moves the detector and style scores by the same points for
every model, and scales time, requests and tokens by the same factor. --design full measures every
combination. --repeats measures each one more than once; a rewrite's detector and style scores vary
from one draw to the next, and the report gives that spread.

Trials are saved to --out as they finish, so a run that stops (a spent quota, Ctrl-C) resumes where
it left off, and a model whose quota runs out is skipped for the rest of the run. Every rewrite is
also logged as labelled AI data to data/essays/tuning_redteam.jsonl. The table goes to --table
(models/generation_tuning.json), which the Red team page's priorities read, and a report to --report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from eval import generate_essays
from eval.generate_essays import (
    GEMINI_API_URL,
    RETRYABLE_STATUS,
    APIError,
    QuotaExhausted,
    clean_essay,
    load_style_profile,
    style_instructions,
    thinking_level,
)
from eval.generation_settings import EVEN, MAX_POINTS, MEASURES, REFERENCE_BLOCKS, TABLE, choose, scores
from eval.redteam_batch import GROUP_SCHEMA, block_prompt, group_prompt, parse_group_reply
from opendetector.config import DATA_ROOT
from opendetector.documents import extract_document, rewritable_blocks
from opendetector.env import load_env
from opendetector.segment import count_words

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite",
          "gemini-3-flash-preview")  # the Red team page's model and fallbacks
LEVELS = ("minimal", "low", "medium", "high")
GROUPS = (1, 4, 8)
TOKEN_FIELDS = ("total_tokens", "total_input_tokens", "total_cached_tokens", "total_output_tokens",
                "total_thought_tokens")
LABELLED = DATA_ROOT / "data/essays/tuning_redteam.jsonl"
TRIES = 6  # per request, for a server that is busy or briefly unreachable
DAILY_LIMIT = re.compile(r"limit: ([\d,]+) requests per day")


# -- style match ----------------------------------------------------------------------


@dataclass
class StyleYardstick:
    """How typical a text is of the writer whose style profile it follows.

    The profile's folder (data/styles/alexandria/ for alexandria_profile.json) holds every paragraph
    the writer has written, corpus.jsonl, and the measurements update_style_profile.py makes of them.
    A text is measured the same way (sentence lengths and their spread, commas, vocabulary, spelling
    and punctuation habits, banned phrases), and its distance from the profile is the root mean square
    of how far each measurement is off, in units of how much it varies between the writer's own
    passages of the same length. The style match is the share of those passages that are further
    from the profile than the text: 50% is as typical as the writer's median passage."""

    target: dict
    spread: dict[str, float]
    passages: list[float]
    banned: dict

    @classmethod
    def build(cls, profile_path: Path, words: int) -> StyleYardstick | None:
        """None when the profile has no corpus beside it, or too little of one to compare against."""
        import update_style_profile as usp

        folder = profile_path.parent / profile_path.stem.removesuffix("_profile")
        corpus = folder / "corpus.jsonl"
        if not corpus.exists():
            return None
        banned = usp.banned_patterns(json.loads(profile_path.read_text(encoding="utf-8")))
        paragraphs = [entry["text"] for entry in usp.load_corpus(corpus) if "superseded" not in entry]
        chunks, current = [], []
        for paragraph in paragraphs:
            current.append(paragraph)
            if sum(count_words(text) for text in current) >= words:
                chunks.append(current)
                current = []
        summaries = [usp.summarize(usp.measure(chunk, banned)) for chunk in chunks]
        summaries = [summary for summary in summaries if summary]
        if len(summaries) < 5:
            return None
        try:
            target = json.loads((folder / "measurements.json").read_text(encoding="utf-8"))["stacked"]
        except (OSError, ValueError, KeyError):
            target = usp.summarize(usp.measure(paragraphs, banned))
        spread = {}
        for label, _ in _measures():
            values = [value for summary in summaries if (value := _value(summary, label)) is not None]
            if len(values) >= 5 and statistics.pstdev(values) > 0:
                spread[label] = statistics.pstdev(values)
        yardstick = cls(target, spread, [], banned)
        distances = (yardstick.distance(summary) for summary in summaries)
        yardstick.passages = [distance for distance in distances if distance is not None]
        return yardstick if len(yardstick.passages) >= 5 else None

    def distance(self, summary: dict) -> float | None:
        offsets = []
        for label, spread in self.spread.items():
            value, target = _value(summary, label), _value(self.target, label)
            if value is not None and target is not None:
                offsets.append(((value - target) / spread) ** 2)
        return math.sqrt(statistics.fmean(offsets)) if offsets else None

    def match(self, text: str) -> float | None:
        """The style match of `text`, 0 to 100."""
        import update_style_profile as usp

        paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
        summary = usp.summarize(usp.measure(paragraphs, self.banned))
        distance = summary and self.distance(summary)
        if distance is None:
            return None
        further = sum(1.0 if other > distance else 0.5 if other == distance else 0.0 for other in self.passages)
        return round(100 * further / len(self.passages), 1)


def _measures() -> list[tuple[str, tuple[str, ...]]]:
    import update_style_profile as usp

    return [*usp.DRIFT, ("banned phrases per 1000", ())]


def _value(summary: dict, label: str) -> float | None:
    """One measurement from an update_style_profile summary."""
    if label == "banned phrases per 1000":
        words = summary.get("words")
        return 1000 * sum((summary.get("banned_phrases_used") or {}).values()) / words if words else None
    value = summary
    for key in dict(_measures())[label]:
        value = value.get(key) if isinstance(value, dict) else None
    return float(value) if isinstance(value, int | float) else None


# -- trials ---------------------------------------------------------------------------


def sample_jobs(name: str, data: bytes, words: int, min_blocks: int = 1) -> tuple[list[dict], SimpleNamespace]:
    """The run of blocks every trial rewrites, from the first, until it holds `words` words and at
    least `min_blocks` blocks, and a plan like the Red team page's for the prompts: every block,
    matching each one's length."""
    sections, blocks = extract_document(name, data), rewritable_blocks(name, data)
    kind = "markdown_block" if name.lower().endswith(".md") else "text_block"
    jobs, total = [], 0
    for index, block in enumerate(blocks, 1):
        if total >= words and len(jobs) >= min_blocks:
            break
        jobs.append({"kind": kind, "label": block.label, "question": "", "previous": block.text, "target": block,
                     "key": f"{index:03d} {kind} {block.label}"})
        total += count_words(block.text)
    plan = SimpleNamespace(prompt="", context_text="\n\n".join(f"[{s.label}]\n{s.text}" for s in sections),
                           blocks=blocks, answer_slots=[], match_existing_lengths=True, answer_words=100)
    return jobs, plan


def planned_trials(models: list[str], levels: list[str], groups: list[int], repeats: int, design: str,
                   baseline: str) -> list[dict]:
    """Every trial to run: each model at each of its thinking levels (a model without a level is
    asked for the next one up, so it is measured once) with the largest group, and either every group
    size with those, or every group size with the baseline model at low thinking."""
    largest = max(groups)
    combos = {}
    for model in models:
        for level in dict.fromkeys(thinking_level(model, level) for level in levels):
            for group in (groups if design == "full" else [largest]):
                combos[(model, level, group)] = None
    if design == "main":
        for group in groups:
            combos[(baseline, thinking_level(baseline, "low"), group)] = None
    return [{"model": model, "thinking": level, "group_size": group, "repeat": repeat,
             "key": f"{model}|{level}|{group}|{repeat}"}
            for (model, level, group) in combos for repeat in range(1, repeats + 1)]


def _transient(exc: APIError) -> bool:
    message = str(exc)
    status = re.search(r"HTTP (\d{3})", message)
    return (int(status.group(1)) in RETRYABLE_STATUS if status
            else any(text in message for text in ("could not reach", "empty response")))


def run_trial(trial: dict, jobs: list[dict], plan, system: str, api_key: str, temperature: float) -> dict:
    """Rewrite the sample run with one setting, as a batch would: a request per group, then any block
    its reply leaves out on its own. Raises QuotaExhausted, or APIError for a request that keeps failing."""
    model, thinking, size = trial["model"], trial["thinking"], trial["group_size"]
    seconds, requests, retried, tokens, texts = 0.0, 0, 0, Counter(), {}

    def ask(prompt: str, **options) -> str:
        nonlocal seconds, requests, retried
        for attempt in range(TRIES):
            usage: dict = {}
            started = time.monotonic()
            try:
                reply = generate_essays.chat(GEMINI_API_URL, api_key, model, prompt, temperature, retries=1,
                                             system=system, thinking=thinking, usage=usage, **options)
            except QuotaExhausted:
                raise
            except APIError as exc:
                if not _transient(exc) or attempt == TRIES - 1:
                    raise
                retried += 1
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            seconds += time.monotonic() - started
            requests += 1
            tokens.update({field: usage.get(field, 0) for field in TOKEN_FIELDS})
            return reply
        raise AssertionError("unreachable")

    left_out = 0
    for start in range(0, len(jobs), size):
        group = jobs[start:start + size]
        alone = group
        if len(group) > 1:
            reply = ask(group_prompt(plan, group), json_schema=GROUP_SCHEMA, timeout=60 + 15 * len(group))
            parsed = parse_group_reply(reply, [f"B{number}" for number in range(1, len(group) + 1)])
            alone = []
            for number, job in enumerate(group, 1):
                if text := clean_essay(parsed.get(f"B{number}", "")):
                    texts[job["key"]] = text
                else:
                    alone.append(job)
            left_out += len(alone)
        for job in alone:
            texts[job["key"]] = clean_essay(ask(block_prompt(plan, job), timeout=60))
    return {**trial, "blocks": len(jobs), "text": "\n\n".join(texts[job["key"]] for job in jobs),
            "seconds": round(seconds, 2), "requests": requests, "retried": retried, "left_out": left_out,
            "tokens": dict(tokens)}


# -- fitting --------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def aggregate(trials: list[dict]) -> dict:
    """One measured setting from its trials."""
    blocks = sum(trial["blocks"] for trial in trials)
    ai = [trial["ai"] for trial in trials if trial.get("ai") is not None]
    style = [trial["style"] for trial in trials if trial.get("style") is not None]
    tokens = sum(trial["tokens"].get("total_tokens") or sum(trial["tokens"].get(field, 0) for field in TOKEN_FIELDS[1:])
                 for trial in trials)
    return {"model": trials[0]["model"], "thinking": trials[0]["thinking"], "group_size": trials[0]["group_size"],
            "ai": _mean(ai), "ai_sd": _sd(ai), "style": _mean(style), "style_sd": _sd(style),
            "seconds_per_block": sum(trial["seconds"] for trial in trials) / blocks,
            "requests_per_block": sum(trial["requests"] for trial in trials) / blocks,
            "tokens_per_block": tokens / blocks, "trials": len(trials), "measured": True}


def fit(trials: list[dict], baseline: str) -> list[dict]:
    """Every measured setting, and every other combination of a measured model and thinking level
    with a measured group size, predicted by adding the group size's effect on the baseline model."""
    by_setting: dict[tuple, list[dict]] = {}
    for trial in trials:
        by_setting.setdefault((trial["model"], trial["thinking"], trial["group_size"]), []).append(trial)
    measured = {setting: aggregate(group) for setting, group in by_setting.items()}
    rows = dict(measured)
    largest = max((group for _, _, group in measured), default=0)
    base = {group: row for (model, level, group), row in measured.items()
            if model == baseline and level == thinking_level(baseline, "low")}
    reference = base.get(largest)
    for (model, level, group), anchor in measured.items():
        if group != largest or reference is None:
            continue
        for other, effect in base.items():
            if (model, level, other) not in rows:
                rows[(model, level, other)] = _predicted(anchor, effect, reference, other)
    return sorted(rows.values(), key=lambda row: (row["model"], LEVELS.index(row["thinking"])
                                                   if row["thinking"] in LEVELS else 9, row["group_size"]))


def _predicted(anchor: dict, effect: dict, reference: dict, group: int) -> dict:
    """The anchor's setting with another group size: the baseline model's change from its own
    reference group to that size, added to the detector and style scores and multiplied into the rest."""
    def shifted(name: str, low: float, high: float) -> float | None:
        values = (anchor[name], effect[name], reference[name])
        return None if None in values else min(high, max(low, values[0] + values[1] - values[2]))

    def scaled(name: str) -> float:
        return anchor[name] * effect[name] / reference[name] if reference[name] else anchor[name]

    return {"model": anchor["model"], "thinking": anchor["thinking"], "group_size": group,
            "ai": shifted("ai", 0.0, 1.0), "ai_sd": None, "style": shifted("style", 0.0, 100.0), "style_sd": None,
            "seconds_per_block": scaled("seconds_per_block"), "requests_per_block": scaled("requests_per_block"),
            "tokens_per_block": scaled("tokens_per_block"), "trials": 0, "measured": False}


def _pick(table: dict, measure: str | None) -> dict:
    points = EVEN if measure is None else {m: MAX_POINTS if m == measure else 0 for m in MEASURES}
    setting = choose(table, points)
    return {"model": setting.model, "thinking": setting.thinking, "group_size": setting.group_size,
            "parallel": setting.parallel}


def report(table: dict) -> str:
    """The table as Markdown: the default, what each measure picks on its own, and every setting."""
    rows, limits = table["rows"], table.get("daily_requests") or {}
    even = {m: 1 / len(MEASURES) for m in MEASURES}
    utility = [sum(even[m] * score[m] for m in MEASURES) for score in scores(rows, REFERENCE_BLOCKS, limits)]
    known = ", ".join(f"{model} {limit:,}" for model, limit in sorted(limits.items())) or "none yet"

    def setting(pick: dict) -> str:
        return f"{pick['model']}, {pick['thinking']} thinking, {pick['group_size']} blocks per request"

    def number(value: float | None, sd: float | None, scale: float = 1, digits: int = 0) -> str:
        if value is None:
            return "n/a"
        return f"{value * scale:.{digits}f}" + (f" ± {sd * scale:.{digits}f}" if sd is not None else "")

    lines = [
        "# Generation settings", "",
        f"Measured {table['created']} on {table['blocks']} blocks ({table['words']} words) of `{table['document']}`, "
        f"with the `{table['style_profile']}` style profile and the {table['detector_profile']} detector. "
        "Written by `python -m eval.tune_generation`.", "",
        f"**Default (points spread evenly):** {setting(table['default'])}, {table['default']['parallel']} requests "
        "at once.", "",
        "All points on one measure:", "",
        *(f"- {measure}: {setting(_pick(table, measure))}" for measure in MEASURES), "",
        "AI likelihood is the detector's, lower is better. Style match is the share of the writer's own "
        "passages that are further from the profile, higher is better. ± is the spread between repeats: "
        "differences smaller than it are noise. Predicted rows add the group size's effect measured on "
        f"{table['baseline_model']}.", "",
        f"Free requests a day, from Gemini's quota errors or set by hand: {known}. A model whose limit isn't "
        f"known loses no quota score for it. The picks and scores here are for a {REFERENCE_BLOCKS}-block "
        "batch; the Red team page scores the document it has.", "",
        "| Model | Thinking | Blocks per request | AI likelihood % | Style match % | Seconds per block "
        "| Requests per block | Tokens per block | Trials | Even-points score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, value in sorted(zip(rows, utility, strict=True), key=lambda pair: -pair[1]):
        lines.append(
            f"| {row['model']} | {row['thinking']} | {row['group_size']} | {number(row['ai'], row['ai_sd'], 100)} "
            f"| {number(row['style'], row['style_sd'])} | {row['seconds_per_block']:.1f} "
            f"| {row['requests_per_block']:.2f} | {row['tokens_per_block']:,.0f} "
            f"| {row['trials'] or 'predicted'} | {value:.2f} |")
    return "\n".join(lines) + "\n"


# -- the run --------------------------------------------------------------------------


def _load_trials(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_trials(path: Path, trials: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(trial, ensure_ascii=False) + "\n" for trial in trials), encoding="utf-8")
    os.replace(temporary, path)


def _log_labelled(trial: dict, document: str, source: str, style: str) -> None:
    """Every rewrite is labelled AI evaluation data, like the Red team page's."""
    LABELLED.parent.mkdir(parents=True, exist_ok=True)
    row = {"id": f"tuning-{source}-{trial['key'].replace('|', '-')}", "text": trial["text"], "label": 1,
           "dataset": "local_tuning", "domain": "document_combined", "model": trial["model"],
           "attack": f"full_document+{style}", "group": "ai", "group_key": f"document-{source}",
           "thinking": trial["thinking"], "group_size": trial["group_size"], "source_file": document}
    with LABELLED.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _limits_path(trials_path: Path) -> Path:
    return trials_path.with_name(trials_path.stem + "_limits.json")


def _load_limits(path: Path) -> dict[str, int]:
    try:
        return {str(model): int(limit) for model, limit in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_limits(path: Path, limits: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(limits.items())), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document", nargs="?", help=".md, .docx or .pptx to rewrite")
    parser.add_argument("--style", type=Path, default=ROOT / "data/styles/alexandria_profile.json",
                        help="style profile sent as the system prompt and measured against")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--thinking", default=",".join(LEVELS))
    parser.add_argument("--groups", default=",".join(map(str, GROUPS)), help="blocks per request to try")
    parser.add_argument("--design", choices=("main", "full"), default="main")
    parser.add_argument("--baseline-model", default="gemini-3.5-flash-lite",
                        help="the model the group sizes are measured with in the main design")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--words", type=int, default=250, help="how much of the document each trial rewrites")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--profile", choices=("full", "slim"), default="slim", help="the detector build")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/cache/tuning.jsonl")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/tuning.md")
    parser.add_argument("--fit-only", action="store_true", help="skip new trials; fit what --out holds")
    parser.add_argument("--daily-limit", action="append", default=[], metavar="MODEL=N",
                        help="a model's free requests a day, when its quota errors haven't said; repeatable")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows as it happens, also in a log file
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    levels = [name.strip() for name in args.thinking.split(",") if name.strip()]
    groups = sorted({int(size) for size in args.groups.split(",")})
    if unknown := set(levels) - set(LEVELS):
        parser.error(f"unknown thinking levels: {', '.join(sorted(unknown))}")
    limits_path = _limits_path(args.out)
    limits = _load_limits(limits_path)
    for entry in args.daily_limit:
        model, _, limit = entry.partition("=")
        if not model.strip() or not limit.strip().isdigit():
            parser.error(f"--daily-limit takes MODEL=N, not {entry!r}")
        limits[model.strip()] = int(limit)
    if args.daily_limit:
        _save_limits(limits_path, limits)

    trials = _load_trials(args.out)
    meta = trials[0].get("meta", {}) if trials else {}
    if not args.fit_only:
        if not args.document:
            parser.error("name a document to measure on, or pass --fit-only")
        load_env()
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(f"${args.api_key_env} is not set")
        path = Path(args.document)
        data = path.read_bytes()
        # Two requests or more at the largest group size, so every size measured is one a batch would send.
        jobs, plan = sample_jobs(path.name, data, args.words, min_blocks=2 * max(groups))
        system = style_instructions(load_style_profile(args.style))
        source = hashlib.sha256(data).hexdigest()[:16]
        meta = {"document": path.name, "source": source, "style_profile": args.style.stem, "words": args.words,
                "sample_words": sum(count_words(job["previous"]) for job in jobs), "blocks": len(jobs),
                "context_chars": len(system) + len(plan.context_text), "detector_profile": args.profile,
                "baseline_model": args.baseline_model, "temperature": args.temperature}
        same = ("source", "style_profile", "words", "blocks")
        kept = [trial for trial in trials if all(trial.get("meta", {}).get(key) == meta[key] for key in same)]
        if len(kept) < len(trials):
            print(f"Leaving out {len(trials) - len(kept)} trials measured on another document, profile or length.")
        trials, done = kept, {trial["key"] for trial in kept}
        planned = planned_trials(models, levels, groups, args.repeats, args.design, args.baseline_model)
        skipped: set[str] = set()
        print(f"{len(planned) - len(done & {t['key'] for t in planned})} of {len(planned)} trials to run, "
              f"{len(jobs)} blocks each.")
        for trial in planned:
            if trial["key"] in done or trial["model"] in skipped:
                continue
            try:
                result = run_trial(trial, jobs, plan, system, api_key, args.temperature)
            except QuotaExhausted as exc:
                print(f"  {trial['model']} is out of quota; skipping its other trials this run. {exc}", file=sys.stderr)
                skipped.add(trial["model"])
                if match := DAILY_LIMIT.search(str(exc)):
                    limits[trial["model"]] = int(match.group(1).replace(",", ""))
                    _save_limits(limits_path, limits)
                continue
            except APIError as exc:
                print(f"  {trial['model']} failed; skipping its other trials this run. {exc}", file=sys.stderr)
                skipped.add(trial["model"])
                continue
            trials.append({**result, "meta": meta})
            _save_trials(args.out, trials)
            _log_labelled(result, path.name, source, meta["style_profile"])
            print(f"  {trial['key']}: {result['requests']} requests, {result['seconds']:.0f} s")

        unscored = [trial for trial in trials if "ai" not in trial]
        if unscored:
            from opendetector import Detector, DetectorConfig

            print(f"Scoring {len(unscored)} rewrites with the {args.profile} detector…")
            detector = Detector.from_config(DetectorConfig.for_profile(args.profile))
            yardstick = StyleYardstick.build(args.style, meta["sample_words"])
            for trial in unscored:
                trial["ai"] = detector.analyze(trial["text"], sentences=False).ai_probability
                trial["style"] = yardstick.match(trial["text"]) if yardstick else None
                _save_trials(args.out, trials)

    if not trials:
        print("No trials yet.")
        return 1
    if args.fit_only:
        meta = trials[0]["meta"]
    scored = [trial for trial in trials if "ai" in trial]
    table = {"created": time.strftime("%Y-%m-%d"), **{key: meta[key] for key in (
        "document", "blocks", "context_chars", "detector_profile", "style_profile", "baseline_model")},
        "words": meta["sample_words"], "rows": fit(scored, meta["baseline_model"]), "daily_requests": limits}
    table["default"] = _pick(table, None)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(table), encoding="utf-8")
    print(f"Wrote {args.table} ({len(table['rows'])} settings) and {args.report}. Default: "
          f"{table['default']['model']}, {table['default']['thinking']} thinking, "
          f"{table['default']['group_size']} blocks per request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
