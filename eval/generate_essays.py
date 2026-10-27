"""Write AI "mirrors" of human essays: same assignment, similar length, different author.

Pairing each human essay with an AI essay on the same prompt means the detector is judged
on style, not topic. Defaults to Gemini through its OpenAI-compatible endpoint, with the key
read from GEMINI_API_KEY (or the project's .env). Any OpenAI-compatible endpoint works
(OpenAI, OpenRouter, a local LM Studio or Ollama server) via --base-url and --api-key-env,
so you can cover many model families.

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --style student --out data/essays/mirrors_gemini_student.jsonl

Styles: plain (default), student (asks for a student's voice), humanize (writes, then asks
the model to rewrite the essay so it evades AI detectors — the attack you need to resist).

--style-profile adds a JSON style profile (a writer's measured rhythm, vocabulary, banned
phrases and mechanical slips) as the system prompt, so the mirrors imitate one writer:

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --style-profile data/styles/student_profile.json --out data/essays/mirrors_profile.jsonl

The profile's file name joins the attack label (plain+student_profile), so the report scores
style-imitating essays separately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from opendetector.env import load_env

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-3.8-flash"
# The free tier allows very few requests on the newest Flash model (20 when this was written);
# Flash-Lite's allowance is larger, so it is the fallback when the main model's quota runs out.
FALLBACK_MODEL = "gemini-3.5-flash-lite"

PROMPTS = {
    "plain": "Write an essay responding to the assignment below. Aim for about {words} words. "
             "Reply with the essay only.\n\nAssignment:\n{assignment}",
    "student": "You are a grade {grade} student. Write your essay for the assignment below in your own "
               "natural voice, about {words} words. Reply with the essay only.\n\nAssignment:\n{assignment}",
}
HUMANIZE = ("Rewrite this essay so it reads as if a real student wrote it and would not be flagged by AI "
            "detectors. Keep the content and length. Reply with the essay only.\n\n{essay}")
STYLE_INTRO = ("Write in the style of the writer profiled below. Follow every rule exactly. The spelling slips, "
               "missing apostrophes, stray capitals and spacing quirks are part of the style: keep them, don't "
               "correct them. The reference section is the measured style the rules come from.")


def load_style_profile(path: Path) -> dict:
    """A style profile as a dict. The top-level "style_profile" wrapper is optional."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data.get("style_profile", data)


def _inline(items: list) -> bool:
    """Numbers (a range) and long lists of short phrases (banned words) read best on one line;
    anything else, rules above all, stays a bulleted list."""
    if all(isinstance(x, (int, float)) for x in items):
        return True
    return len(items) >= 10 and all(isinstance(x, str) and len(x) <= 40 for x in items)


def _render(value, depth: int) -> list[str]:
    """Nested profile sections as Markdown under their JSON keys, which the rules refer to by
    name ("any word in banned_ai_cliches"). A section's plain values and inline lists come
    first, one line each, then a heading per subsection or bulleted list."""
    if isinstance(value, list):
        return [f"- {json.dumps(x) if isinstance(x, (dict, list)) else x}" for x in value]
    if not isinstance(value, dict):
        return [str(value)]
    lines, sections = [], []
    for key, item in value.items():
        if isinstance(item, list) and _inline(item):
            lines.append(f"{key}: {', '.join(map(str, item))}")
        elif isinstance(item, (dict, list)):
            sections += ["", f"{'#' * depth} {key}", *_render(item, depth + 1)]
        else:
            lines.append(f"{key}: {item}")
    return lines + sections


def style_instructions(profile: dict) -> str:
    """The system prompt for a style profile: the overall rule, then the hard rules, then the
    measurements as reference. Profiles without system_prompt_overrides are all reference."""
    rules = dict(profile.get("system_prompt_overrides") or {})
    golden = rules.pop("golden_transformation_rule", None)
    reference = {k: v for k, v in profile.items() if k != "system_prompt_overrides"}
    lines = [STYLE_INTRO]
    if golden:
        lines += ["", golden]
    if rules:
        lines += ["", "# Rules", *_render(rules, 2)]
    if reference:
        lines += ["", "# Reference", *_render(reference, 2)]
    return "\n".join(lines)


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_WAIT = 120  # seconds; a server asking us to wait longer than this means a spent quota
_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")


class APIError(RuntimeError):
    pass


class QuotaExhausted(APIError):
    pass


def is_local(base_url: str) -> bool:
    return urllib.parse.urlparse(base_url).hostname in ("localhost", "127.0.0.1", "::1")


def _error_details(exc: urllib.error.HTTPError) -> tuple[str, float | None]:
    """The server's error message, and how many seconds it asks us to wait (None if it doesn't say)."""
    try:
        data = json.loads(exc.read() or b"null")
    except ValueError:
        data = None
    if isinstance(data, list) and data:  # Gemini's OpenAI-compatible endpoint wraps errors in a list
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    message = (error.get("message") if isinstance(error, dict) else error) or str(exc.reason)
    message = " ".join(str(message).split())

    retry_after = (exc.headers.get("retry-after") or "").strip()
    if retry_after.isdigit():
        return message, float(retry_after)
    match = _RETRY_IN.search(message)
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g or 0) for g in match.groups())
        return message, hours * 3600 + minutes * 60 + seconds
    return message, None


def chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5,
         system: str | None = None, timeout: int = 300) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content or not content.strip():
                raise APIError(f"empty response from {model}: {json.dumps(data)[:300]}")
            return content.strip()
        except urllib.error.HTTPError as exc:
            message, delay = _error_details(exc)
            if exc.code == 429 and delay is not None and delay > MAX_WAIT:
                raise QuotaExhausted(f"{model} is out of quota: {message}") from exc
            # Bad requests, unknown models and rejected keys will never succeed; don't retry them.
            if exc.code not in RETRYABLE_STATUS or attempt == retries - 1:
                raise APIError(f"HTTP {exc.code} from {base_url}: {message}") from exc
            reason, wait = f"HTTP {exc.code}", delay if delay is not None else 2 ** attempt * 5
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise APIError(f"could not reach {base_url}: {getattr(exc, 'reason', exc)}") from exc
            reason, wait = str(getattr(exc, "reason", exc)), 2 ** attempt * 5
        print(f"  request failed ({reason}); retrying in {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _grade(value) -> int:
    """PERSUADE grade levels arrive as floats (8.0) when the column has blanks; default to 10."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 10


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persuade", required=True, help="PERSUADE 2.0 CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL,
                        help="switch to this model when --model is out of quota ('' to just stop)")
    parser.add_argument("--style", choices=["plain", "student", "humanize"], default="plain")
    parser.add_argument("--style-profile", type=Path,
                        help="JSON style profile to write in, sent as the system prompt; its file name is "
                             "added to the attack label (<style>+<name>)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-url", default=GEMINI_OPENAI_URL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="environment variable holding the key")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.persuade).is_file():
        parser.error(f"{args.persuade} not found. Download PERSUADE 2.0 from "
                     "github.com/scrosseye/persuade_corpus_2.0 first.")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not is_local(args.base_url):
        parser.error(f"${args.api_key_env} is not set. Export your key for {args.base_url}, point "
                     "--api-key-env at the variable that holds it, or use a local --base-url.")
    system, attack = None, args.style
    if args.style_profile:
        try:
            system = style_instructions(load_style_profile(args.style_profile))
        except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
            parser.error(f"could not read the style profile {args.style_profile}: {e}")
        attack = f"{args.style}+{args.style_profile.stem}"

    df = pd.read_csv(args.persuade)
    text_col = next((c for c in ("full_text", "text") if c in df), None)
    id_col = next((c for c in ("essay_id_comp", "essay_id", "id") if c in df), None)
    if text_col is None or id_col is None:
        parser.error(f"{args.persuade} needs an essay text column (full_text or text) and an id column "
                     f"(essay_id_comp, essay_id or id); found {list(df.columns)}")
    df = df.sample(frac=1, random_state=args.seed).head(args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    template = PROMPTS["student" if args.style == "student" else "plain"]

    def write_essay(model: str, prompt: str) -> str:
        essay = chat(args.base_url, api_key, model, prompt, args.temperature, system=system)
        if args.style == "humanize":
            essay = chat(args.base_url, api_key, model, HUMANIZE.format(essay=essay), args.temperature,
                         system=system)
        return essay

    model, written = args.model, 0
    with out.open("a") as f:
        for _, row in df.iterrows():
            source_id = str(row[id_col])
            assignment = row.get("assignment")
            prompt = template.format(
                assignment=assignment if isinstance(assignment, str) else row.get("prompt_name", "an essay"),
                words=max(150, round(len(str(row[text_col]).split()), -1)),
                grade=_grade(row.get("grade_level")),
            )
            essay = None
            while f"{source_id}-{model}-{attack}" not in done:
                try:
                    essay = write_essay(model, prompt)
                    break
                except QuotaExhausted as e:
                    if not args.fallback_model or model == args.fallback_model:
                        print(f"Stopped after {written} new essays: {e}", file=sys.stderr)
                        return 1
                    print(f"{e}\nSwitching to {args.fallback_model} for the rest of the run.", file=sys.stderr)
                    model = args.fallback_model
                except APIError as e:
                    print(f"Stopped after {written} new essays: {e}", file=sys.stderr)
                    return 1
            if essay is None:
                continue  # already written in an earlier run
            essay_id = f"{source_id}-{model}-{attack}"
            f.write(json.dumps({
                "id": essay_id, "text": essay, "label": 1, "dataset": "persuade_mirror",
                "domain": row.get("prompt_name", "essay"), "model": model, "attack": attack,
                "group": "ai", "group_key": source_id,
            }) + "\n")
            f.flush()
            written += 1
            print(f"  wrote {essay_id}")
    print(f"Wrote {written} new essays to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
