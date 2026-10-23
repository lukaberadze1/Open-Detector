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
import time
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


def chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # network errors and rate limits: back off and retry
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
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

    api_key = os.environ.get(args.api_key_env, "")
    df = pd.read_csv(args.persuade).sample(frac=1, random_state=args.seed).head(args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()

    with out.open("a") as f:
        for _, row in df.iterrows():
            source_id = str(row.get("essay_id_comp", row.get("essay_id")))
            essay_id = f"{source_id}-{args.model}-{args.style}"
            if essay_id in done:
                continue
            fields = {
                "assignment": row.get("assignment") if isinstance(row.get("assignment"), str) else row.get("prompt_name"),
                "words": max(150, round(len(str(row["full_text"]).split()), -1)),
                "grade": row.get("grade_level", 10),
            }
            template = PROMPTS["student" if args.style == "student" else "plain"]
            essay = chat(args.base_url, api_key, args.model, template.format(**fields), args.temperature)
            if args.style == "humanize":
                essay = chat(args.base_url, api_key, args.model, HUMANIZE.format(essay=essay), args.temperature)
            f.write(json.dumps({
                "id": essay_id, "text": essay, "label": 1, "dataset": "persuade_mirror",
                "domain": row.get("prompt_name", "essay"), "model": args.model, "attack": args.style,
                "group": "ai", "group_key": source_id,
            }) + "\n")
            f.flush()
            print(f"  wrote {essay_id}")


if __name__ == "__main__":
    main()
