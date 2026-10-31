"""Measure standard generation: new samples written from a prompt alone, in a style profile.

    python -m eval.tune_prompts --style data/styles/alexandria_profile.json --register finalized

eval/tune_generation.py measures rewriting a document; this measures the Adversarial Lab's other way of
writing, a new piece from instructions, with the same prompt the page sends (redteam_batch.sample_prompt).
Each prompt of a set is written by each model, `--repeats` times, and each sample is scored on its own:
the detector's AI likelihood and verdict, and its style match against the profile's writing in the
register. `--framing`, `--plan` and `--old-voice` are the variants measured on 2026-10-30:
- `--framing challenge` names the piece an "AI-generated adversarial challenge sample", as the page
  first did; `task` (the default) asks for the piece itself.
- `--plan` gives the sample a whole sentence plan borrowed from the profile's writer (borrowed_plan).
- `--old-voice` adds the rule the page first sent with no source text, which forbade "I" and "you".
`--revise N` then rewrites the passages the detector flags, up to N rounds (redteam_batch.revise_flagged),
and records the AI likelihood after each round; `--examples N` sends N of the writer's own passages.

Samples are resumable in `--out` and logged as labelled AI data, like every other generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from eval import generate_essays
from eval.generate_essays import (
    GEMINI_API_URL,
    APIError,
    QuotaExhausted,
    clean_essay,
    load_style_profile,
    profile_registers,
    style_instructions,
    thinking_level,
)
from eval.redteam_batch import (DRAFT_STRETCH, FRAMINGS, GROUP_SCHEMA, borrowed_plan, restyle, revise_flagged,
                                sample_prompt)
from eval.tune_generation import LABELLED, TOKEN_FIELDS, StyleYardstick, style_examples, writer_shapes
from opendetector.env import load_env
from opendetector.segment import count_words

ROOT = Path(__file__).resolve().parents[1]
# The rule the page sent with a prompt alone until 2026-10-30, for --old-voice.
OLD_VOICE = ("Match the grammatical person used in the source text; do not add first- or second-person language "
             "if it is absent.")
# Prompts a writer might be given, finished work and quick, informal writing.
PROMPTS = {
    "finished": ["An essay for a history class: why did the Roman Republic fall?",
                 "A report for a PE class on how sleep affects athletic performance.",
                 "A short story about someone's first day at a new job."],
    "informal": ["A review of a pizza place you went to last weekend, for a review site.",
                 "A reply on a forum to someone who says homework should be banned.",
                 "An explanation, for a friend, of how compound interest works."],
}


def run_sample(model: str, prompt: str, system: str | None, api_key: str, temperature: float,
               thinking: str, shapes: list | None = None, analyze=None, rounds: int = 0) -> dict:
    """One sample: a draft, then with `shapes` that draft rewritten in the writer's sentences (restyle),
    then with `rounds` its flagged passages revised against the detector's `analyze` (revise_flagged)."""
    tokens, started = dict.fromkeys(TOKEN_FIELDS, 0), time.monotonic()

    def ask(request: str, jobs: list[dict] | None = None) -> str:
        usage: dict = {}
        reply = generate_essays.chat(GEMINI_API_URL, api_key, model, request, temperature, retries=3, system=system,
                                     thinking=thinking_level(model, thinking), usage=usage,
                                     **({"json_schema": GROUP_SCHEMA} if jobs is not None else {}))
        for field in TOKEN_FIELDS:
            tokens[field] += usage.get(field, 0)
        return reply

    draft = clean_essay(ask(prompt))
    text = restyle(draft, shapes, ask, clean_essay) if shapes else draft
    history = None
    if rounds:
        text, history = revise_flagged(text, analyze, ask, clean_essay, shapes or [], rounds)
    return {"text": text, "draft": draft if shapes else None, "seconds": round(time.monotonic() - started, 2),
            "tokens": tokens, "rounds": history}


def summary(samples: list[dict]) -> list[str]:
    lines = ["| Model | Samples | Before revising % | AI likelihood % | Likely human | No verdict | Style match % "
             "| Words | Tokens per sample |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in sorted({sample["model"] for sample in samples}):
        mine = [sample for sample in samples if sample["model"] == model and "verdict" in sample]
        if not mine:
            continue
        ai = [100 * sample["ai"] for sample in mine if sample["ai"] is not None]  # None: the detector abstained
        style = [sample["style"] for sample in mine if sample.get("style") is not None]
        human = sum(sample["verdict"] == "likely_human" for sample in mine)
        before = [100 * sample["rounds"][0] for sample in mine
                  if sample.get("rounds") and sample["rounds"][0] is not None]
        lines.append(f"| {model} | {len(mine)} "
                     f"| {f'{statistics.fmean(before):.0f} ± {statistics.pstdev(before):.0f}' if before else 'n/a'} "
                     f"| {f'{statistics.fmean(ai):.0f} ± {statistics.pstdev(ai):.0f}' if ai else 'n/a'} "
                     f"| {human} of {len(mine)} | {len(mine) - len(ai)} "
                     f"| {f'{statistics.fmean(style):.0f}' if style else 'n/a'} "
                     f"| {statistics.fmean(sample['words'] for sample in mine):.0f} "
                     f"| {statistics.fmean(sample['tokens'].get('total_tokens', 0) for sample in mine):,.0f} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", type=Path, default=ROOT / "styles/human_default_profile.json")
    parser.add_argument("--register", help="the profile's register to write in (default: its first)")
    parser.add_argument("--no-style", action="store_true", help="write without the profile, measured against it")
    parser.add_argument("--set", choices=list(PROMPTS), default="finished", help="which prompts to write")
    parser.add_argument("--framing", choices=FRAMINGS, default="task")
    parser.add_argument("--plan", action="store_true", help="a sentence plan borrowed from the writer")
    parser.add_argument("--old-voice", action="store_true", help="the first page's rule forbidding I and you")
    parser.add_argument("--restyle", action="store_true",
                        help="rewrite each draft passage by passage in the writer's sentences, as the page does")
    parser.add_argument("--revise", type=int, default=0, metavar="N",
                        help="rewrite the passages the detector flags, up to N rounds")
    parser.add_argument("--examples", type=int, default=0, metavar="N", help="send N of the writer's own passages")
    parser.add_argument("--models", default="gemini-3.5-flash-lite")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--words", type=int, default=350)
    parser.add_argument("--profile", choices=("full", "slim"), default="slim", help="the detector build")
    parser.add_argument("--out", type=Path, required=True, help="samples, resumable")
    args = parser.parse_args(argv)

    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        parser.error("$GEMINI_API_KEY is not set")
    profile = load_style_profile(args.style)
    registers = profile_registers(profile)
    register = (args.register if args.register in registers else registers[0]) if registers else None
    examples, checklist = ([], []) if args.no_style or not args.examples else \
        style_examples(args.style, register, args.examples)
    system = None if args.no_style else style_instructions(profile, register, examples, checklist)
    shapes = writer_shapes(args.style, register) if args.plan or args.restyle else []
    label = ("plain" if args.no_style else f"{args.style.stem}:{register}" if register else args.style.stem) + \
        f"+{args.framing}" + ("+plan" if args.plan else "") + ("+old-voice" if args.old_voice else "") + \
        ("+restyle" if args.restyle else "") + \
        ("+examples" if examples else "") + (f"+revise{args.revise}" if args.revise else "")
    voice = OLD_VOICE if args.old_voice else ""
    done = {}
    if args.out.exists():
        done = {sample["key"]: sample for sample in map(json.loads, args.out.read_text().splitlines()) if sample}
    samples = list(done.values())
    detector = None
    if args.revise:
        from opendetector import Detector, DetectorConfig

        detector = Detector.from_config(DetectorConfig.for_profile(args.profile))
    for model in (name.strip() for name in args.models.split(",") if name.strip()):
        for number, material in enumerate(PROMPTS[args.set], 1):
            for repeat in range(1, args.repeats + 1):
                key = f"{label}|{model}|{args.set}{number}|{repeat}"
                if key in done:
                    continue
                plan = borrowed_plan(shapes, args.words, key) if args.plan else None
                words = round(args.words * DRAFT_STRETCH) if args.restyle else args.words
                prompt = sample_prompt(material, words, voice, args.framing, plan)
                try:
                    sample = run_sample(model, prompt, system, api_key, args.temperature, args.thinking,
                                        shapes if args.restyle else None,
                                        detector.analyze if detector else None, args.revise)
                except (QuotaExhausted, APIError) as exc:
                    print(f"  {model} failed, skipping it: {exc}")
                    break
                except (OSError, ValueError) as exc:  # a dropped connection or a garbled reply: the next sample
                    print(f"  {key} failed, going on: {exc!r}")
                    continue
                sample.update(key=key, label=label, model=model, prompt=material, repeat=repeat,
                              words=count_words(sample["text"]))
                samples.append(sample)
                with args.out.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"  {key}: {sample['words']} words, {sample['seconds']:.0f} s")
    unscored = [sample for sample in samples if "ai" not in sample]
    if unscored:
        from opendetector import Detector, DetectorConfig

        detector, sticks = detector or Detector.from_config(DetectorConfig.for_profile(args.profile)), {}
        for sample in unscored:
            report = detector.analyze(sample["text"], sentences=False)
            size = max(50, round(sample["words"] / 50) * 50)
            if size not in sticks:
                sticks[size] = StyleYardstick.build(args.style, size, register)
            sample.update(ai=report.ai_probability, verdict=report.verdict,
                          style=sticks[size].match(sample["text"]) if sticks[size] else None)
        args.out.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
        LABELLED.parent.mkdir(parents=True, exist_ok=True)
        with LABELLED.open("a", encoding="utf-8") as output:
            for sample in unscored:
                output.write(json.dumps({
                    "id": "prompt-" + hashlib.sha256(sample["key"].encode()).hexdigest()[:16], "text": sample["text"],
                    "label": 1, "dataset": "local_tuning", "domain": "custom_prompt", "model": sample["model"],
                    "attack": f"prompt+{label}", "group": "ai", "group_key": f"prompt-{sample['prompt']}",
                    "prompt": sample["prompt"], "temperature": args.temperature, "thinking": args.thinking,
                }, ensure_ascii=False) + "\n")
    print("\n".join([f"\n{label}, {args.set} prompts:", *summary([s for s in samples if s["label"] == label])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
