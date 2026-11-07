"""Measures revising a whole-document rewrite against the detector, with the page's prompts and settings.

Each repeat writes one rewrite of a document's opening blocks (tune_generation.run_trial), then revises it
(redteam_batch.revise_blocks), by sentence scores or by the hottest document window. `--as-page` revises as
the Red team page does: keeping names and numbers, and dropping what leaves the style profile. Several
documents make it a pattern rather than one text's result: `--mage N` writes N AI-written documents from
MAGE (other domains and generators, none in the evaluation's cached sample) beside the given folder. Every
text is logged as labelled AI data. Resumable (`--out`), and it waits out an overloaded model. Spends quota.
Writes `--report`, read again with no new runs by `--report-only`; `--full` also scores the finished texts
with the full detector, which the revision never saw.

    python -m eval.tune_revise --mage 8 data/essays/sources/revise
    python -m eval.tune_revise data/essays/sources/claude_salamis.md data/essays/sources/revise/*.md --as-page
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from eval import generate_essays
from eval.generate_essays import (PROVIDERS, APIError, QuotaExhausted, load_style_profile, style_instructions,
                                  styled_clean)
from eval.redteam_batch import (ANCHORED_CANDIDATES, GROUP_SCHEMA, REVISE, REVISE_CANDIDATES, REVISE_KEEPING,
                                WIN_BELOW, chunk_scorer, document_windows, revise_anchored, revise_blocks, style_guard)
from eval.trace import Trace, text_id
from eval.text_attacks import (
    CLAUSES,
    FREE_TRIP,
    LOCAL_PIVOTS,
    PIVOTS,
    attack_label,
    attack_name,
    local_attack_name,
    rewrite_attack_name,
    run_attacks,
)
from eval.tune_generation import StyleYardstick, _log_labelled, run_trial, sample_jobs, writer_shapes
from opendetector.combiner import HUMAN_CUTOFF
from opendetector.env import load_env
from opendetector.metrics import wilson_interval
from opendetector.segment import count_words, split_sentences

ROOT = Path(__file__).resolve().parents[1]
GROUP = 8
QUOTA_EXIT = 3  # so a runner can tell a spent quota from a crash
DOWN_EXIT = 4  # and a model that stays overloaded from either
UNUSABLE_EXIT = 5  # and one that refuses the request itself: too large, unknown model, rejected key
_REFUSED = re.compile(r"HTTP (400|401|403|404|413)\b")
GIVE_UPS = 2  # repeats in a row given up on before calling the model down: each has already waited out BUSY_WAITS
METHODS = ("sentences", "windows")
BUSY_WAITS = (60, 120, 300, 300)  # seconds before writing a repeat again after a model is overloaded or times out
# Instructions tried on top of the style profile, by name (--note); each is labelled in the attack.
NOTES = {
    "tells": "Leave out the habits that give AI writing away (from github.com/blader/humanizer): 'not X but Y', a "
             "short closing line that repeats the point, a staged run-up before the point, things grouped in threes "
             "when the meaning has no three parts, stacked qualifiers, inflated importance, sales language, a "
             "trailing '-ing' clause, stock AI words, and comments on the writing itself. Where the style profile "
             "says otherwise, it wins.",
    "active": "Prefer verbs to abstract nouns: write 'they decided' over 'they made a decision', and use the "
              "active voice wherever the sentence allows it.",
    "asymmetric": "Avoid even, balanced writing: don't give every point the same weight or length, skip the "
                  "introduction-and-summary shape, and let some points get one clause and others several sentences.",
    "punctuation": "Vary your punctuation unevenly: some sentences with no commas at all, some with several, an "
                   "occasional dash or colon, and straight or curly quotes as they come.",
    "parenthetical": "Now and then put a brief aside of your own in brackets, a passing reaction to the point being "
                     "made. Add no facts that aren't in the original.",
    "interjection": "Once or twice a passage, break in with a short rhetorical question or a blunt remark of a few "
                    "words, as someone thinking aloud would. Add no facts that aren't in the original.",
    "idiom": "Use an everyday idiom or turn of phrase where it fits naturally, the kind people say out loud, and "
             "avoid stock literary ones.",
    "digression": "Let a sentence wander a little now and then: a small qualifier, a self-correction such as 'or "
                  "rather' or 'at least in most cases', or a minor side remark. Add no facts that aren't in the "
                  "original.",
}
_COMMON = {"The", "This", "That", "These", "Those", "There", "They", "Their", "When", "After", "While", "Although",
           "However", "From", "With", "What", "Then", "Some", "Many", "Most", "Such", "Also", "Introduction",
           "Conclusion", "Background", "Overall", "Furthermore", "Moreover", "Additionally", "Finally"}


def fact_terms(text: str) -> set[str]:
    """What a rewrite should still say: the names (capitalized words, sentence openers mostly left out by
    `_COMMON`) and the numbers of 2 digits or more."""
    return set(re.findall(r"\b(?:[A-Z][a-z]{3,}|\d{2,})\b", text)) - _COMMON


def facts_kept(source: str, text: str) -> float | None:
    """The share of the source's names and numbers still in `text`, as a percentage."""
    terms = fact_terms(source)
    return round(100 * sum(term in text for term in terms) / len(terms), 1) if terms else None


def mage_documents(folder: Path, count: int, seed: int = 11, words: int = 400, score_above: float = 0.8,
                   human: bool = False) -> list[Path]:
    """Up to `count` AI-written documents from MAGE as Markdown files, one per domain and generator before any
    repeats. Each is one the slim detector reads as AI, and none is in the evaluation's cached sample. MAGE keeps
    no paragraph breaks, so sentences are grouped into paragraphs of three. With `human`, human-written ones the
    detector reads as human (at or under HUMAN_CUTOFF) instead: the page's own case, a writer regenerating their
    work, where a revision can start from the original's wording."""
    from eval.datasets import load_mage
    from opendetector import Detector, DetectorConfig

    cached = ROOT / "eval/cache/mage.jsonl"
    used = {json.loads(line)["id"] for line in cached.open(encoding="utf-8")} if cached.exists() else set()
    used |= {path.stem for path in (ROOT / "data/essays/sources").rglob("*.md")}  # no document twice
    frame = load_mage("test", n=20000, seed=seed)
    frame = frame[(frame["label"] == (0 if human else 1)) & (frame["attack"] == "none") & ~frame["id"].isin(used)]
    frame = frame[frame["text"].map(count_words) >= words]
    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    folder.mkdir(parents=True, exist_ok=True)
    written, domains, generators = [], set(), set()
    for fresh in (True, False):
        for _, row in frame.iterrows():
            if len(written) >= count:
                return written
            if row["id"] in {path.stem for path in written}:
                continue
            if fresh and (row["domain"] in domains or row["model"] in generators):
                continue
            score = detector.score(row["text"])
            if (score > HUMAN_CUTOFF) if human else (score < score_above):
                continue
            sentences = [row["text"][s.start:s.end].strip() for s in split_sentences(row["text"])]
            body = "\n\n".join(" ".join(sentences[at:at + 3]) for at in range(0, len(sentences), 3))
            path = folder / f"{row['id']}.md"
            path.write_text(body + "\n", encoding="utf-8")
            written.append(path)
            domains.add(row["domain"])
            generators.add(row["model"])
            print(f"{path.name}: {row['domain']}, {row['model']}, {count_words(body)} words", flush=True)
    return written


def report(rows: list[dict], method: str) -> str:
    """The report over documents: each document's runs, then all of them. Runs of one document aren't independent,
    so the per-document means count as much as the overall interval."""
    done = [row for row in rows if method in row]
    means = {}
    for row in done:
        means.setdefault(row["document"], []).append(row)
    lines = ["# Revising rewrites against the detector", "",
             f"{len(done)} runs on {len(means)} AI-written documents, revising by {method}. The bar is an AI "
             f"likelihood under {WIN_BELOW:.0%}. Written by `python -m eval.tune_revise`. Brackets are Wilson 95% "
             "intervals, which are too narrow in as far as runs of one document resemble each other.", "",
             "| Document | Runs | Written % | Revised % | Range % | Under the bar | Full detector % | Style distance "
             "before > after | Names, numbers kept % | Length |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def line(name: str, group: list[dict]) -> str:
        ends = [row[method]["history"][-1] for row in group]
        full = [row[method]["full"] for row in group if row[method].get("full") is not None]
        before = [row["distance"] for row in group if row.get("distance") is not None]
        after = [row[method]["distance"] for row in group if row[method].get("distance") is not None]
        facts = [row[method]["facts"] for row in group if row[method].get("facts") is not None]
        start = sum(row["start"] for row in group) / len(group)
        return (f"| {name} | {len(group)} | {100 * start:.0f} | {100 * statistics.mean(ends):.0f} | "
                f"{100 * min(ends):.0f}-{100 * max(ends):.0f} | {sum(end < WIN_BELOW for end in ends)} | "
                + (f"{100 * statistics.mean(full):.0f}" if full else "-") + " | "
                + (f"{statistics.mean(before):.2f} > {statistics.mean(after):.2f}" if before and after else "-") + " | "
                + (f"{statistics.mean(facts):.0f}" if facts else "-") + " | "
                f"x{statistics.mean(row[method]['words'] / row['words'] for row in group):.2f} |")

    lines += [line(name, group) for name, group in sorted(means.items())]
    if done:
        ends = [row[method]["history"][-1] for row in done]
        low, high = wilson_interval(sum(end < WIN_BELOW for end in ends), len(ends))
        doc_means = [statistics.mean(row[method]["history"][-1] for row in group) for group in means.values()]
        lines += [line("**All**", done), "",
                  f"- **Under the bar:** {sum(end < WIN_BELOW for end in ends)} of {len(ends)} runs "
                  f"({low:.0%} to {high:.0%}).",
                  f"- **Documents whose mean is under the bar:** {sum(mean < WIN_BELOW for mean in doc_means)} of "
                  f"{len(doc_means)}.",
                  f"- **Median run:** {100 * statistics.median(ends):.1f}%."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", type=Path, nargs="*")
    parser.add_argument("--mage", type=int, metavar="N",
                        help="write N documents from MAGE to the folder given, and stop")
    parser.add_argument("--tag", help="added to the setup label, to keep these runs apart, e.g. human-sources")
    parser.add_argument("--human-sources", action="store_true",
                        help="with --mage, human-written documents the detector reads as human, not AI-written ones")
    parser.add_argument("--style", type=Path, default=ROOT / "data/styles/alexandria_profile.json")
    parser.add_argument("--register", default="finalized")
    parser.add_argument("--structure", default="writer")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="gemini",
                        help="whose API --model is on; its key comes from .env unless --api-key-env says otherwise")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--words", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10, help="at most: revising stops when it stalls")
    parser.add_argument("--methods", default="sentences",
                        help="how to revise: sentences or windows (revise_blocks), or anchored (revise_anchored, as "
                             "the page does now: every flagged passage a round, half the tries from the original)")
    parser.add_argument("--candidates", type=int, default=REVISE_CANDIDATES,
                        help="rewrites a passage gets each round, the best kept: the page's three use three times "
                             "the requests")
    parser.add_argument("--drafts", type=int, default=1,
                        help="write this many first drafts and revise the lowest-scoring")
    parser.add_argument("--trace", type=Path, help="record every version, rewrite and request here (eval/trace.py)")
    parser.add_argument("--no-log", action="store_true", help="save no labelled data: for testing this script")
    parser.add_argument("--note", choices=list(NOTES), action="append", default=[],
                        help="an instruction added to the style profile; repeat it to stack them")
    parser.add_argument("--mask-facts", action="store_true",
                        help="send names and numbers as placeholders, in writing and revising (eval/fact_mask.py)")
    parser.add_argument("--no-guard", action="store_true",
                        help="with --as-page, let a revision leave the style profile, for one with no writer to match")
    parser.add_argument("--as-page", action="store_true",
                        help="revise as the page does: keep names and numbers, and stay close to the style profile")
    parser.add_argument("--attack", action="append", default=[],
                        choices=[CLAUSES, FREE_TRIP, *(attack_name(pivot) for pivot in PIVOTS),
                                 *(local_attack_name(pivot) for pivot in LOCAL_PIVOTS),
                                 *(rewrite_attack_name(pivot) for pivot in LOCAL_PIVOTS)],
                        help="run on each rewrite before revising (eval/text_attacks.py); repeat it to stack them")
    parser.add_argument("--top-p", type=float, help="the models' top-p, unset by default")
    parser.add_argument("--top-k", type=int, help="the models' top-k, unset by default")
    parser.add_argument("--api-key-env")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/cache/tune_revise.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/revise.md")
    parser.add_argument("--report-only", action="store_true", help="write the report from what --out holds")
    parser.add_argument("--full", action="store_true", help="also score the finished texts with the full detector")
    args = parser.parse_args(argv)

    if args.mage:
        if len(args.documents) != 1:
            parser.error("--mage takes one folder to write to")
        return 0 if mage_documents(args.documents[0], args.mage, human=args.human_sources) else 1
    load_env()
    style = f"{args.style.stem}:{args.register}+{args.structure}+passages+patience"  # as revise_blocks now does
    system = style_instructions(load_style_profile(args.style), args.register)
    for note in args.note:
        system += f"\n\n{NOTES[note]}"
        style += f"+{note}"
    style += f"+t{args.temperature:g}" + ("+as-page" if args.as_page else "") + ("+no-guard" if args.no_guard else "")
    style += "+masked" if args.mask_facts else ""
    sampling = {option: value for option, value in (("top_p", args.top_p), ("top_k", args.top_k)) if value is not None}
    style += f"+best{args.candidates}" if args.rounds else "+draft"
    style += f"+drafts{args.drafts}" if args.drafts > 1 else ""
    style += attack_label(args.attack) + "".join(f"+{option}{value:g}" for option, value in sampling.items())
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if unknown := set(methods) - {"sentences", "windows", "anchored"}:
        parser.error(f"unknown --methods: {', '.join(sorted(unknown))}")
    style += "+anchored" if methods == ["anchored"] else ""  # a different loop: its runs are cached apart
    style += f"+{args.tag}" if args.tag else ""
    saved = [json.loads(line) for line in args.out.open(encoding="utf-8")] if args.out.exists() else []
    rows = [row for row in saved if row["style"] == style and row["model"] == args.model and "document" in row]

    if not args.report_only:
        provider = PROVIDERS[args.provider]
        args.api_key_env = args.api_key_env or provider.key_env
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(f"${args.api_key_env} is not set")
        from opendetector import Detector, DetectorConfig

        detector = Detector.from_config(DetectorConfig.for_profile("slim"))
        shapes = writer_shapes(args.style, args.register)
        clean = styled_clean(args.style, args.register, swaps=True)

        trace = Trace(args.trace, setup=style, model=args.model) if args.trace else None
        if trace:
            trace.watch_requests()
        spent = Counter()  # requests after the first drafts, which run_trial counts: the price of revising

        def ask(prompt: str, pending: list[dict]) -> str:
            spent["revise"] += 1
            return generate_essays.chat(provider.base_url, api_key, args.model, prompt, args.temperature, retries=2,
                                        system=system, thinking=args.thinking, json_schema=GROUP_SCHEMA,
                                        timeout=60 + 15 * len(pending), sampling=sampling or None)

        def ask_plain(prompt: str) -> str:
            spent["attack"] += 1
            return generate_essays.chat(provider.base_url, api_key, args.model, prompt, args.temperature, retries=2,
                                        thinking=args.thinking, timeout=180, sampling=sampling or None)

        given_up = 0
        for path in args.documents:
            data = path.read_bytes()
            source = hashlib.sha256(data).hexdigest()[:16]
            jobs, plan = sample_jobs(path.name, data, args.words, min_blocks=2 * GROUP, structure=args.structure)
            plan.shapes, plan.clean, plan.mask_facts = shapes, clean, args.mask_facts
            original = data.decode("utf-8", "replace")

            def write_repeat(trial: dict, jobs=jobs, plan=plan, path=path, source=source,
                             original=original) -> tuple[dict, list[tuple[dict, str]]]:
                """One repeat's scores, and its texts to log, kept until it is whole so a retry logs nothing twice."""
                spent.clear()
                began = time.monotonic()
                if trace:
                    trace.bind(run=trial["key"], document=path.name, repeat=trial["repeat"],
                               thinking=args.thinking, temperature=args.temperature)
                    trace.version("source", original, detector.analyze(original, sentences=False))
                drafts = [run_trial({**trial, "key": f"{trial['key']}-draft{number}"} if number else trial, jobs, plan,
                                    system, api_key, args.temperature, provider.base_url, sampling or None)
                          for number in range(args.drafts)]
                reports = [detector.analyze(draft["text"], sentences=False) for draft in drafts]
                draft_scores = [report.ai_probability for report in reports]
                result = drafts[draft_scores.index(min(draft_scores))]
                if trace:
                    parent = trace.head
                    for draft, report in zip(drafts, reports, strict=True):
                        trace.head = parent
                        trace.version("draft", draft["text"], report, keep=False, requests=draft["requests"],
                                      chosen=draft is result, words_kept=draft.get("words_kept"),
                                      structure_match=draft.get("structure_match"))
                    trace.head = text_id(result["text"])
                logs = [(draft, style) for draft in drafts]
                pieces = result["text"].split("\n\n")
                attacked, changes = run_attacks(pieces, args.attack, ask_plain, clean)
                if trace and args.attack:
                    trace.version("attack", "\n\n".join(attacked), attacks=args.attack, changes=changes)
                parts = {f"{number:03d}": part for number, part in enumerate(attacked, 1)}
                yardstick = StyleYardstick.build(args.style, count_words(result["text"]), args.register)
                row = {"document": path.name, "source": source, "style": style, "model": args.model,
                       "repeat": trial["repeat"], "words": count_words(result["text"]), "draft": result["text"],
                       "start": min(draft_scores), "draft_scores": draft_scores,
                       "requests": sum(draft["requests"] for draft in drafts),
                       "copied": result["copied"], "words_kept": result["words_kept"],
                       "distance": yardstick.text_distance(result["text"]) if yardstick else None,
                       "facts": facts_kept(original, result["text"]), "attack_changes": changes}
                for method in methods:
                    guarded = args.as_page and yardstick and not args.no_guard
                    fits = style_guard(yardstick.text_distance, result["text"]) if guarded else None
                    outcomes = Counter()
                    if method == "anchored":  # the blocks as written, each against the writer's original of it
                        blocks = result.get("texts") or {}
                        if args.attack or not blocks:  # attacks rework the paragraphs: no block to match it to
                            blocks, originals = parts, {}
                        else:
                            originals = {job["key"]: job["previous"] for job in jobs}
                        from eval.math_attack import CachedScorer

                        changed, history = revise_anchored(
                            blocks, originals, chunk_scorer(detector), CachedScorer(detector), ask, clean, shapes,
                            args.rounds, prompt=REVISE_KEEPING, facts=True, candidates=ANCHORED_CANDIDATES,
                            fits=fits, stats=outcomes, mask=args.mask_facts, trace=trace)
                        text = "\n\n".join(changed.get(key, part) for key, part in blocks.items())
                        row[method] = {"history": history, "changed": len(changed), "words": count_words(text),
                                       "outcomes": dict(outcomes), "text": text,
                                       "distance": yardstick.text_distance(text) if yardstick else None,
                                       "facts": facts_kept(original, text)}
                        logs.append(({**result, "key": f"{trial['key']}-{method}", "text": text},
                                     f"{style}+revise{args.rounds}-{method}"))
                        continue
                    changed, history = revise_blocks(
                        parts, detector.analyze, ask, clean, shapes, args.rounds, fits=fits,
                        windows=document_windows(detector) if method == "windows" else None,
                        prompt=REVISE_KEEPING if args.as_page else REVISE, facts=args.as_page,
                        candidates=args.candidates, stats=outcomes, mask=args.mask_facts,
                        trace=trace)
                    # A run with --rounds 0 and nothing more measures the attack alone.
                    text = "\n\n".join(changed.get(key, part) for key, part in parts.items())
                    row[method] = {"history": history, "changed": len(changed), "words": count_words(text),
                                   "outcomes": dict(outcomes),
                                   "text": text,
                                   "distance": yardstick.text_distance(text) if yardstick else None,
                                   "facts": facts_kept(original, text)}
                    logs.append(({**result, "key": f"{trial['key']}-{method}", "text": text},
                                 f"{style}+revise{args.rounds}-{method}"))
                if trace:
                    for method in methods:
                        final = row[method]
                        trace.version("final", final["text"], method=method, score=final["history"][-1],
                                      facts=final["facts"], distance=final["distance"], outcomes=final["outcomes"])
                row["revise_requests"], row["attack_requests"] = spent["revise"], spent["attack"]
                row["seconds"] = round(time.monotonic() - began)
                return row, logs

            done = {row["repeat"] for row in rows if row["source"] == source}
            for repeat in range(1, args.repeats + 1):
                if repeat in done:
                    continue
                variant = "+".join(args.note) or "base"
                variant += ("-as-page" if args.as_page else "") + ("-no-guard" if args.no_guard else "")
                key = f"{args.style.stem}-{variant}-{args.model}-{args.thinking}-{GROUP}-{args.temperature:g}-{repeat}"
                # Each setup its own ids: the saved-sample reports keep one row per id.
                key += "-" + hashlib.sha256(style.encode()).hexdigest()[:8]
                trial = {"key": f"revise-{key}",
                         "model": args.model, "thinking": args.thinking, "group_size": GROUP,
                         "temperature": args.temperature, "repeat": repeat}
                waits = iter(BUSY_WAITS)
                while True:
                    try:
                        row, logs = write_repeat(trial)
                        break
                    except QuotaExhausted as exc:
                        print(f"Out of quota at {path.name} repeat {repeat}: {exc}", file=sys.stderr)
                        if rows:
                            args.report.write_text(report(rows, methods[0]), encoding="utf-8")
                        return QUOTA_EXIT
                    except APIError as exc:
                        if _REFUSED.search(str(exc)):  # waiting can't fix it: Groq's 413 cost 13 minutes a repeat
                            print(f"{args.model} refused the request: {str(exc)[:200]}", file=sys.stderr)
                            if rows:
                                args.report.write_text(report(rows, methods[0]), encoding="utf-8")
                            return UNUSABLE_EXIT
                        wait = next(waits, None)
                        if wait is None:
                            print(f"Giving up on {path.name} repeat {repeat}: {exc}", file=sys.stderr)
                            row = None
                            given_up += 1
                            break
                        print(f"{path.name} repeat {repeat} failed, again in {wait} s: {str(exc)[:80]}", flush=True)
                        time.sleep(wait)
                if row is None:
                    if given_up < GIVE_UPS:
                        continue
                    print(f"{args.model} looks down: {GIVE_UPS} repeats in a row gave up", file=sys.stderr)
                    return DOWN_EXIT
                given_up = 0
                for logged, label in [] if args.no_log else logs:
                    _log_labelled(logged, path.name, source, label)
                rows.append(row)
                with args.out.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row) + "\n")
                print(f"{path.name} {repeat}: start {row['start']:.3f}, " + ", ".join(
                    f"{method} {' > '.join(f'{p:.3f}' for p in row[method]['history'])} "
                    f"({row[method]['changed']} blocks, facts {row['facts']} > {row[method]['facts']})"
                    for method in methods), flush=True)

    if args.full and rows:
        from opendetector import Detector, DetectorConfig

        detector = Detector.from_config(DetectorConfig.for_profile("full"))
        others = [row for row in saved if row not in rows]
        for row in rows:
            for method in methods:
                if method in row and "full" not in row[method]:
                    row[method]["full"] = detector.analyze(row[method]["text"], sentences=False).ai_probability
        args.out.write_text("".join(json.dumps(row) + "\n" for row in others + rows), encoding="utf-8")
    if rows:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report(rows, methods[0]), encoding="utf-8")
        print(report(rows, methods[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
