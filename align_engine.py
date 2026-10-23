"""Red-team harness: generate text with Claude, then rewrite it until the local ensemble
scores it below a target AI likelihood.

Detector-guided rewriting is the strongest attack a detector faces, so this is how you
find out how robust the ensemble really is. Every draft is saved (with --log) as a
labelled AI sample, so evasions can be scored with `python -m eval.featurize local` and
used as hard examples when you train your own model.

    python align_engine.py "Write a 600-word essay on loneliness in Of Mice and Men" \\
        --log data/attacks/mice_and_men.jsonl --out final.txt

Needs Anthropic credentials (ANTHROPIC_API_KEY, or `ant auth login`) and loads every
detector engine, so don't run it alongside a featurize job on an 8 GB machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import anthropic

from opendetector import Detector, DetectorConfig, Report

# The original spec named claude-3-5-sonnet-20241022, which was retired in October 2025.
MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

REWRITE_PROMPT = """Here is a draft you wrote:

<draft>
{draft}
</draft>

An AI-text detector flagged these passages as machine-written:

{passages}

Rewrite the draft. In the flagged passages, vary the sentence-length cadence: mix short, plain \
declarations with longer, detailed breakdowns, and avoid uniform or repeated phrasing. Keep the \
content, meaning and overall length. Reply with the full revised text only."""

log = logging.getLogger("align_engine")


class GenerationError(RuntimeError):
    pass


class MissingCredentialsError(GenerationError):
    pass


def complete(client: anthropic.Anthropic, model: str, prompt: str) -> str:
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",  # on a policy decline, the API retries on a recommended fallback model
            messages=[{"role": "user", "content": prompt}],
        )
    except TypeError as e:
        # With no credentials at all, the SDK raises a bare TypeError before sending anything,
        # not AuthenticationError (that one means a key was sent and rejected).
        if "authentication method" not in str(e):
            raise
        raise MissingCredentialsError(
            "no Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."
        ) from e
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise GenerationError(f"the model declined the request (category: {category})")
    if response.stop_reason == "max_tokens":
        log.warning("Response hit max_tokens and is truncated")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise GenerationError(f"empty response (stop_reason: {response.stop_reason})")
    return text


def flagged_segments(report: Report, max_segments: int = 4) -> list[str]:
    """The passages to rewrite: runs of consecutive sentences above the detector's review
    band, highest-scoring first. If nothing crosses the band, the three top sentences."""
    review = report.thresholds["review"]
    runs, current = [], []
    for s in report.sentences:
        if s.ai_probability > review:
            current.append(s)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        runs = [[s] for s in sorted(report.sentences, key=lambda s: -s.ai_probability)[:3]]
    runs.sort(key=lambda run: -max(s.ai_probability for s in run))
    return [report.text[run[0].start : run[-1].end] for run in runs[:max_segments]]


def _record(log_path: Path | None, run_id: str, group_key: str, attempt: int, text: str,
            score: float | None, model: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "id": f"{run_id}-{attempt}", "text": text, "label": 1, "dataset": "adaptive_attack",
            "domain": "generated", "model": model, "attack": f"rewrite_{attempt}", "group": "ai",
            "group_key": group_key, "score": score,
        }) + "\n")


def generate_aligned_text(
    prompt: str,
    target_threshold: float = 0.05,
    max_attempts: int = 4,
    *,
    client: anthropic.Anthropic | None = None,
    detector: Detector | None = None,
    model: str = MODEL,
    log_path: Path | None = None,
    device: str = "auto",
) -> str:
    """Generate a completion and rewrite its most AI-like passages until the detector's AI
    likelihood is below `target_threshold`. Returns the first draft that gets there, or the
    lowest-scoring draft after `max_attempts` drafts."""
    client = client or anthropic.Anthropic()
    group_key = hashlib.sha1(prompt.encode()).hexdigest()[:10]
    run_id = f"{group_key}-{int(time.time())}"

    best_text, best_score = "", float("inf")
    draft = complete(client, model, prompt)
    # Load the models only once the API has answered, so a credentials problem fails fast.
    detector = detector or Detector.from_config(DetectorConfig(device=device))
    for attempt in range(max_attempts):
        report = detector.analyze(draft)
        _record(log_path, run_id, group_key, attempt, draft, report.ai_probability, model)
        if report.ai_probability is None:
            log.warning("Draft is %d words; the detector needs %d to score it. Returning it unscored.",
                        report.words, detector.config.min_words)
            return draft

        score = report.ai_probability
        log.info("Attempt %d/%d: AI likelihood %.1f%% (%s%s)", attempt + 1, max_attempts, 100 * score,
                 report.verdict, "" if report.calibrated else ", uncalibrated")
        if score < best_score:
            best_text, best_score = draft, score
        if score < target_threshold:
            log.info("Below the %.1f%% target after %d attempt(s)", 100 * target_threshold, attempt + 1)
            return draft
        if attempt == max_attempts - 1:
            break

        segments = flagged_segments(report)
        log.info("Rewriting %d flagged passage(s)", len(segments))
        passages = "\n\n".join(f"<passage>\n{s}\n</passage>" for s in segments)
        draft = complete(client, model, REWRITE_PROMPT.format(draft=draft, passages=passages))

    log.warning("Never got below %.1f%% in %d attempts; returning the best draft (%.1f%%)",
                100 * target_threshold, max_attempts, 100 * best_score)
    return best_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("prompt", nargs="?", help="the generation prompt")
    source.add_argument("--prompt-file", type=Path, help="read the prompt from a file")
    parser.add_argument("--target", type=float, default=0.05, help="stop below this AI likelihood (default 0.05)")
    parser.add_argument("--max-attempts", type=int, default=4, help="drafts to try, including the first")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--log", type=Path, help="append every draft as a labelled AI sample (JSONL)")
    parser.add_argument("--out", type=Path, help="write the final text here instead of stdout")
    parser.add_argument("--device", default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    if args.max_attempts < 1 or not 0 < args.target < 1:
        parser.error("--max-attempts must be >= 1 and --target between 0 and 1")
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt

    try:
        text = generate_aligned_text(prompt, args.target, args.max_attempts, model=args.model,
                                     log_path=args.log, device=args.device)
    except anthropic.AuthenticationError:
        log.error("The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY or run `ant auth login`.")
        return 1
    except anthropic.NotFoundError:
        log.error("Model %r not found or not available to this key.", args.model)
        return 1
    except anthropic.RateLimitError:
        log.error("Rate limited even after the SDK's retries; try again later.")
        return 1
    except anthropic.APIStatusError as e:
        log.error("API error %s: %s", e.status_code, e.message)
        return 1
    except anthropic.APIConnectionError:
        log.error("Could not reach the Anthropic API; check the network connection.")
        return 1
    except GenerationError as e:
        log.error("Generation failed: %s", e)
        return 1

    if args.out:
        args.out.write_text(text + "\n")
        log.info("Wrote %s", args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
