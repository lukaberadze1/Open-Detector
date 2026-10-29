"""Measure adversarial challenge-generation settings on a document for the Adversarial Lab.

    python -m eval.tune_generation report.docx                 # measure, then write the table
    python -m eval.tune_generation report.docx --models gemini-3.5-flash-lite --repeats 1
    python -m eval.tune_generation --fit-only                  # rewrite the table from the trials so far

A setting is a model, its thinking level and how many blocks go in one request. Each trial rewrites
the same run of the document's blocks (at least --words words) the way a challenge batch does, with the
same prompts and the style profile as the system prompt, and records:

- detector: the AI likelihood of the rewritten run, from the --profile build of the detector
- style: how closely the rewrite follows the style profile (see StyleYardstick); with --no-style the
  rewrite is written without one and still measured against it, the control for the profile's effect
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
(models/generation_tuning.json), which the Adversarial Lab's priorities read, and a report to --report.
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
    profile_registers,
    style_instructions,
    takes_temperature,
    thinking_level,
)
from eval.generation_settings import (
    EVEN,
    LIMITS,
    MAX_POINTS,
    MEASURES,
    REFERENCE_BLOCKS,
    TABLE,
    choose,
    learned_limits,
    limits_in,
    record_limits,
    scores,
)
from eval.redteam_batch import (
    GROUP_SCHEMA,
    block_prompt,
    group_by_words,
    group_prompt,
    parse_group_reply,
    worth_rewriting,
)
from opendetector.config import DATA_ROOT
from opendetector.documents import extract_document, rewritable_blocks
from opendetector.env import load_env
from opendetector.segment import count_words

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite",
          "gemini-3-flash-preview")  # the Adversarial Lab's model and fallbacks
LEVELS = ("minimal", "low", "medium", "high")
GROUPS = (1, 4, 8)
TEMPERATURES = (0.8, 1.0, 1.3)
BASE_TEMPERATURE = 0.8  # the page's default creativity, which the first trials were measured at
TOKEN_FIELDS = ("total_tokens", "total_input_tokens", "total_cached_tokens", "total_output_tokens",
                "total_thought_tokens")
LABELLED = DATA_ROOT / "data/essays/tuning_redteam.jsonl"
TRIES = 6  # per request, for a server that is busy or briefly unreachable


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

    def feedback(self, text: str, limit: int = 3) -> list[str]:
        """Return the largest measurable style deviations as concrete revision directions."""
        import update_style_profile as usp

        paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
        summary = usp.summarize(usp.measure(paragraphs, self.banned))
        if not summary:
            return []
        deviations = []
        for label, _ in _measures():
            value, target, spread = _value(summary, label), _value(self.target, label), self.spread.get(label)
            if value is None or target is None or not spread:
                continue
            z = (value - target) / spread
            if abs(z) >= 0.75:
                direction = "lower" if z > 0 else "higher"
                deviations.append((abs(z), f"Move {label} {direction}, toward the profile's usual level "
                                   f"(this draft: {value:g}; profile: {target:g})."))
        return [instruction for _, instruction in sorted(deviations, reverse=True)[:limit]]


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
    least `min_blocks` blocks, and a plan like the Adversarial Lab's for the prompts: every block,
    matching each one's length."""
    sections, blocks = extract_document(name, data), rewritable_blocks(name, data)
    kind = "markdown_block" if name.lower().endswith(".md") else "text_block"
    jobs, total = [], 0
    for index, block in enumerate(blocks, 1):
        if total >= words and len(jobs) >= min_blocks:
            break
        if not worth_rewriting(block.text):  # left as it is, as a batch leaves it
            continue
        jobs.append({"kind": kind, "label": block.label, "question": "", "previous": block.text, "target": block,
                     "key": f"{index:03d} {kind} {block.label}"})
        total += count_words(block.text)
    plan = SimpleNamespace(prompt="", context_text="\n\n".join(f"[{s.label}]\n{s.text}" for s in sections),
                           blocks=blocks, answer_slots=[], match_existing_lengths=True, answer_words=100)
    return jobs, plan


def trial_key(model: str, level: str, group: int, temperature: float, repeat: int) -> str:
    """A trial's name. Trials at the page's default temperature keep the name they had before
    temperature was measured, so they still count."""
    return f"{model}|{level}|{group}|{repeat}" + ("" if temperature == BASE_TEMPERATURE else f"|t{temperature:g}")


def planned_trials(models: list[str], levels: list[str], groups: list[int], repeats: int, design: str,
                   baseline: str, temperatures: list[float] | None = None,
                   base_temperature: float = 0.8) -> list[dict]:
    """Every trial to run: each model at each of its thinking levels (a model without a level is
    asked for the next one up, so it is measured once) with the largest group at the base temperature,
    and either every group size and temperature with those, or every group size and every temperature
    with the baseline model at low thinking. A model that ignores temperature is measured at one."""
    largest = max(groups)
    temperatures = list(dict.fromkeys([base_temperature, *(temperatures or [])]))
    combos = {}
    for model in models:
        heats = temperatures if design == "full" and takes_temperature(model) else [base_temperature]
        for level in dict.fromkeys(thinking_level(model, level) for level in levels):
            for group in (groups if design == "full" else [largest]):
                for heat in heats:
                    combos[(model, level, group, heat)] = None
    if design == "main":
        low = thinking_level(baseline, "low")
        for group in groups:
            combos[(baseline, low, group, base_temperature)] = None
        for heat in temperatures if takes_temperature(baseline) else []:
            combos[(baseline, low, largest, heat)] = None
    return [{"model": model, "thinking": level, "group_size": group, "temperature": heat, "repeat": repeat,
             "key": trial_key(model, level, group, heat, repeat)}
            for (model, level, group, heat) in combos for repeat in range(1, repeats + 1)]


def _transient(exc: APIError) -> bool:
    message = str(exc)
    status = re.search(r"HTTP (\d{3})", message)
    return (int(status.group(1)) in RETRYABLE_STATUS if status
            else any(text in message for text in ("could not reach", "empty response")))


def run_trial(trial: dict, jobs: list[dict], plan, system: str | None, api_key: str, temperature: float) -> dict:
    """Rewrite the sample run with one setting, as a batch would: a request per group, then any block
    its reply leaves out on its own. Raises QuotaExhausted, or APIError for a request that keeps failing."""
    model, thinking, size = trial["model"], trial["thinking"], trial["group_size"]
    temperature = trial.get("temperature", temperature)
    seconds, requests, retried, tokens, texts = 0.0, 0, 0, Counter(), {}

    def ask(prompt: str, **options) -> str:
        nonlocal seconds, requests, retried
        timeouts = 0
        for attempt in range(TRIES):
            usage: dict = {}
            started = time.monotonic()
            try:
                reply = generate_essays.chat(GEMINI_API_URL, api_key, model, prompt, temperature, retries=1,
                                             system=system, thinking=thinking, usage=usage, **options)
            except QuotaExhausted:
                raise
            except APIError as exc:
                # A model that doesn't answer within the timeout twice is down for this run: waiting out
                # four more timeouts cost 3.8 Flash twelve minutes once.
                timeouts += "timed out" in str(exc)
                if not _transient(exc) or attempt == TRIES - 1 or timeouts >= 2:
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
    by_key = {job["key"]: job for job in jobs}
    for keys in group_by_words(list(by_key), size, lambda key: count_words(by_key[key]["previous"])):
        group = [by_key[key] for key in keys]
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
            "temperature": trials[0].get("temperature", BASE_TEMPERATURE),
            "ai": _mean(ai), "ai_sd": _sd(ai), "style": _mean(style), "style_sd": _sd(style),
            "seconds_per_block": sum(trial["seconds"] for trial in trials) / blocks,
            "requests_per_block": sum(trial["requests"] for trial in trials) / blocks,
            "tokens_per_block": tokens / blocks,
            "input_tokens_per_block": sum(trial["tokens"].get("total_input_tokens", 0) for trial in trials) / blocks,
            "trials": len(trials), "measured": True}


def fit(trials: list[dict], baseline: str, base_temperature: float = BASE_TEMPERATURE) -> list[dict]:
    """Every measured setting, and every other combination of a measured model and thinking level with
    a measured group size and temperature, predicted by adding the baseline model's effect of each:
    its change from the largest group to that size, and from the base temperature to that one. A model
    that ignores temperature gets no temperature but its own."""
    by_setting: dict[tuple, list[dict]] = {}
    for trial in trials:
        setting = (trial["model"], trial["thinking"], trial["group_size"], trial.get("temperature", base_temperature))
        by_setting.setdefault(setting, []).append(trial)
    measured = {setting: aggregate(group) for setting, group in by_setting.items()}
    rows = dict(measured)
    largest = max((group for _, _, group, _ in measured), default=0)
    low = thinking_level(baseline, "low")
    reference = measured.get((baseline, low, largest, base_temperature))
    by_group = {group: row for (model, level, group, heat), row in measured.items()
                if (model, level, heat) == (baseline, low, base_temperature)}
    by_heat = {heat: row for (model, level, group, heat), row in measured.items()
               if (model, level, group) == (baseline, low, largest)}
    for (model, level, group, heat), anchor in measured.items():
        if reference is None or group != largest or heat != base_temperature:
            continue
        heats = by_heat if takes_temperature(model) else {base_temperature: reference}
        for other_group, group_effect in by_group.items():
            for other_heat, heat_effect in heats.items():
                if (model, level, other_group, other_heat) not in rows:
                    rows[(model, level, other_group, other_heat)] = _predicted(
                        anchor, reference, (group_effect, heat_effect), other_group, other_heat)
    return sorted(rows.values(), key=lambda row: (row["model"], LEVELS.index(row["thinking"])
                                                   if row["thinking"] in LEVELS else 9, row["group_size"],
                                                   row["temperature"]))


def _predicted(anchor: dict, reference: dict, effects: tuple[dict, ...], group: int, temperature: float) -> dict:
    """The anchor's setting at another group size and temperature: each of the baseline model's changes
    from its reference setting, added to the detector and style scores and multiplied into the rest."""
    def shifted(name: str, low: float, high: float) -> float | None:
        values = (anchor[name], reference[name], *(effect[name] for effect in effects))
        if None in values:
            return None
        return min(high, max(low, anchor[name] + sum(effect[name] - reference[name] for effect in effects)))

    def scaled(name: str) -> float:
        value = anchor[name]
        for effect in effects:
            value = value * effect[name] / reference[name] if reference[name] else value
        return value

    return {"model": anchor["model"], "thinking": anchor["thinking"], "group_size": group, "temperature": temperature,
            "ai": shifted("ai", 0.0, 1.0), "ai_sd": None, "style": shifted("style", 0.0, 100.0), "style_sd": None,
            "seconds_per_block": scaled("seconds_per_block"), "requests_per_block": scaled("requests_per_block"),
            "tokens_per_block": scaled("tokens_per_block"), "input_tokens_per_block": scaled("input_tokens_per_block"),
            "trials": 0, "measured": False}


def _pick(table: dict, measure: str | None) -> dict:
    points = EVEN if measure is None else {m: MAX_POINTS if m == measure else 0 for m in MEASURES}
    setting = choose(table, points)
    return {"model": setting.model, "thinking": setting.thinking, "group_size": setting.group_size,
            "temperature": setting.temperature, "parallel": setting.parallel}


def report(table: dict) -> str:
    """The table as Markdown: the default, what each measure picks on its own, and every setting."""
    rows, limits = table["rows"], table.get("daily_requests") or {}
    even = {m: 1 / len(MEASURES) for m in MEASURES}
    utility = [sum(even[m] * score[m] for m in MEASURES) for score in scores(rows, REFERENCE_BLOCKS, limits)]
    known = ", ".join(f"{model} {limit:,}" for model, limit in sorted(limits.items())) or "none yet"

    def setting(pick: dict) -> str:
        heat = f", temperature {pick['temperature']:g}" if pick.get("temperature") is not None else ""
        return f"{pick['model']}, {pick['thinking']} thinking{heat}, {pick['group_size']} blocks per request"

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
        "batch; the Adversarial Lab scores the document it has.", "",
        "| Model | Thinking | Temperature | Blocks per request | AI likelihood % | Style match % "
        "| Seconds per block | Requests per block | Tokens per block | Trials | Even-points score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, value in sorted(zip(rows, utility, strict=True), key=lambda pair: -pair[1]):
        lines.append(
            f"| {row['model']} | {row['thinking']} | {row.get('temperature', BASE_TEMPERATURE):g} "
            f"| {row['group_size']} | {number(row['ai'], row['ai_sd'], 100)} "
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
    """Every generated challenge is labelled AI evaluation data, like the Adversarial Lab's."""
    LABELLED.parent.mkdir(parents=True, exist_ok=True)
    row = {"id": f"tuning-{source}-{trial['key'].replace('|', '-')}", "text": trial["text"], "label": 1,
           "dataset": "local_tuning", "domain": "document_combined", "model": trial["model"],
           "attack": f"full_document+{style}", "group": "ai", "group_key": f"document-{source}",
           "thinking": trial["thinking"], "group_size": trial["group_size"],
           "temperature": trial.get("temperature", BASE_TEMPERATURE), "source_file": document}
    with LABELLED.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document", nargs="?", help=".md, .docx or .pptx to rewrite")
    parser.add_argument("--style", type=Path, default=ROOT / "data/styles/alexandria_profile.json",
                        help="style profile sent as the system prompt and measured against")
    parser.add_argument("--register", help="with a profile that has registers, the one to write in (finalized "
                                           "or notes; default: the profile's first)")
    parser.add_argument("--no-style", action="store_true",
                        help="write without a style profile, still measuring against --style: the control a "
                             "profile's effect is measured against")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--thinking", default=",".join(LEVELS))
    parser.add_argument("--groups", default=",".join(map(str, GROUPS)), help="blocks per request to try")
    parser.add_argument("--design", choices=("main", "full"), default="main")
    parser.add_argument("--baseline-model", default="gemini-3.5-flash-lite",
                        help="the model the group sizes are measured with in the main design")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--words", type=int, default=250, help="how much of the document each trial rewrites")
    parser.add_argument("--temperature", type=float, default=BASE_TEMPERATURE,
                        help="the temperature every model is measured at")
    parser.add_argument("--temperatures", default=",".join(f"{t:g}" for t in TEMPERATURES),
                        help="temperatures to measure on the baseline model (every model with --design full)")
    parser.add_argument("--profile", choices=("full", "slim"), default="slim", help="the detector build")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/cache/tuning.jsonl")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/tuning.md")
    parser.add_argument("--fit-only", action="store_true", help="skip new trials; fit what --out holds")
    parser.add_argument("--daily-limit", action="append", default=[], metavar="MODEL=N",
                        help="a model's free requests a day, when its quota errors haven't said; repeatable")
    parser.add_argument("--limits", type=Path, help=f"where daily limits are kept (default {LIMITS}); the Red "
                                                    "team page reads and adds to the same file")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows as it happens, also in a log file
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    levels = [name.strip() for name in args.thinking.split(",") if name.strip()]
    groups = sorted({int(size) for size in args.groups.split(",")})
    temperatures = [float(heat) for heat in args.temperatures.split(",") if heat.strip()]
    if unknown := set(levels) - set(LEVELS):
        parser.error(f"unknown thinking levels: {', '.join(sorted(unknown))}")
    given = {}
    for entry in args.daily_limit:
        model, _, limit = entry.partition("=")
        if not model.strip() or not limit.strip().isdigit():
            parser.error(f"--daily-limit takes MODEL=N, not {entry!r}")
        given[model.strip()] = int(limit)
    record_limits(given, args.limits)

    # Trials measured on another sample stay in the file; only the latest sample's are run and fitted.
    others = _load_trials(args.out)
    meta = others[-1].get("meta", {}) if others else {}
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
        profile = load_style_profile(args.style)
        system = None if args.no_style else style_instructions(profile, args.register)
        source = hashlib.sha256(data).hexdigest()[:16]
        registers = profile_registers(profile)
        register = (args.register if args.register in registers else registers[0]) if registers else None
        written_in = "plain" if args.no_style else f"{args.style.stem}:{register}" if register else args.style.stem
        meta = {"document": path.name, "source": source, "style_profile": written_in, "words": args.words,
                "sample_words": sum(count_words(job["previous"]) for job in jobs), "blocks": len(jobs),
                "context_chars": len(system or "") + len(plan.context_text), "detector_profile": args.profile,
                "baseline_model": args.baseline_model, "temperature": args.temperature}
    same = ("source", "style_profile", "words", "blocks")
    trials = [trial for trial in others if all(trial.get("meta", {}).get(key) == meta.get(key) for key in same)]
    others = [trial for trial in others if trial not in trials]
    if not args.fit_only:
        if others:
            print(f"Leaving out {len(others)} trials measured on another document, profile, length or sample.")
        done = {trial["key"] for trial in trials}
        planned = planned_trials(models, levels, groups, args.repeats, args.design, args.baseline_model,
                                 temperatures, args.temperature)
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
                record_limits(limits_in(f"{trial['model']}: {exc}"), args.limits)
                continue
            except APIError as exc:
                print(f"  {trial['model']} failed; skipping its other trials this run. {exc}", file=sys.stderr)
                skipped.add(trial["model"])
                continue
            trials.append({**result, "meta": meta})
            _save_trials(args.out, others + trials)
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
                _save_trials(args.out, others + trials)

    if not trials:
        print("No trials yet.")
        return 1
    scored = [trial for trial in trials if "ai" in trial]
    rows = fit(scored, meta["baseline_model"], meta.get("temperature", BASE_TEMPERATURE))
    table = {"created": time.strftime("%Y-%m-%d"), **{key: meta[key] for key in (
        "document", "blocks", "context_chars", "detector_profile", "style_profile", "baseline_model")},
        "words": meta["sample_words"], "rows": rows, "daily_requests": learned_limits(args.limits)}
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
