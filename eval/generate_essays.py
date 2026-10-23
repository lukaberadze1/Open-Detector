"""Write AI "mirrors" of human essays: same assignment, similar length, different author.

Pairing each human essay with an AI essay on the same prompt means the detector is judged
on style, not topic. Works with any OpenAI-compatible chat endpoint (OpenAI, OpenRouter,
a local LM Studio or Ollama server), so you can cover many model families.

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --model gpt-5-mini --style student --out data/essays/mirrors_gpt5mini_student.jsonl

Styles: plain (default), student (asks for a student's voice), humanize (writes, then asks
the model to rewrite the essay so it evades AI detectors — the attack you need to resist).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROMPTS = {
    "plain": "Write an essay responding to the assignment below. Aim for about {words} words. "
             "Reply with the essay only.\n\nAssignment:\n{assignment}",
    "student": "You are a grade {grade} student. Write your essay for the assignment below in your own "
               "natural voice, about {words} words. Reply with the essay only.\n\nAssignment:\n{assignment}",
}
HUMANIZE = ("Rewrite this essay so it reads as if a real student wrote it and would not be flagged by AI "
            "detectors. Keep the content and length. Reply with the essay only.\n\n{essay}")


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class APIError(RuntimeError):
    pass


def is_local(base_url: str) -> bool:
    return urllib.parse.urlparse(base_url).hostname in ("localhost", "127.0.0.1", "::1")


def _error_message(exc: urllib.error.HTTPError) -> str:
    try:
        error = json.loads(exc.read() or b"{}").get("error")
        return (error.get("message") if isinstance(error, dict) else error) or exc.reason
    except (ValueError, AttributeError):
        return str(exc.reason)


def chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.load(resp)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content or not content.strip():
                raise APIError(f"empty response from {model}: {json.dumps(data)[:300]}")
            return content.strip()
        except urllib.error.HTTPError as exc:
            # Bad requests, unknown models and rejected keys will never succeed; don't retry them.
            if exc.code not in RETRYABLE_STATUS or attempt == retries - 1:
                raise APIError(f"HTTP {exc.code} from {base_url}: {_error_message(exc)}") from exc
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise APIError(f"could not reach {base_url}: {getattr(exc, 'reason', exc)}") from exc
            reason = str(getattr(exc, "reason", exc))
        wait = 2 ** attempt * 5
        print(f"  request failed ({reason}); retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persuade", required=True, help="PERSUADE 2.0 CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--style", choices=["plain", "student", "humanize"], default="plain")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the key")
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

    written = 0
    with out.open("a") as f:
        for _, row in df.iterrows():
            source_id = str(row[id_col])
            essay_id = f"{source_id}-{args.model}-{args.style}"
            if essay_id in done:
                continue
            assignment, grade = row.get("assignment"), row.get("grade_level")
            fields = {
                "assignment": assignment if isinstance(assignment, str) else row.get("prompt_name", "an essay"),
                "words": max(150, round(len(str(row[text_col]).split()), -1)),
                "grade": int(grade) if pd.notna(grade) and str(grade).isdigit() else 10,
            }
            try:
                essay = chat(args.base_url, api_key, args.model, template.format(**fields), args.temperature)
                if args.style == "humanize":
                    essay = chat(args.base_url, api_key, args.model, HUMANIZE.format(essay=essay), args.temperature)
            except APIError as e:
                print(f"Stopped after {written} new essays: {e}", file=sys.stderr)
                return 1
            f.write(json.dumps({
                "id": essay_id, "text": essay, "label": 1, "dataset": "persuade_mirror",
                "domain": row.get("prompt_name", "essay"), "model": args.model, "attack": args.style,
                "group": "ai", "group_key": source_id,
            }) + "\n")
            f.flush()
            written += 1
            print(f"  wrote {essay_id}")
    print(f"Wrote {written} new essays to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
