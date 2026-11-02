"""Try Groq and OpenRouter with your keys before trusting a run to them: is the key good, what models are on
offer, and does each model write plain text and JSON the way the Red team page needs, with a style profile as
big as a real request's. Each model takes a few short requests; nothing is saved. Prints no keys.

    python -m eval.check_providers --list openrouter
    python -m eval.check_providers groq openrouter
    python -m eval.check_providers groq --models openai/gpt-oss-120b,llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from eval.generate_essays import (
    PROVIDERS,
    APIError,
    chat,
    key_accepted,
    list_models,
    load_style_profile,
    style_instructions,
)
from opendetector.env import load_env

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "styles/human_default_profile.json"
PLAIN = "Write about 80 words on why towns build libraries. Reply with the text alone."
AS_JSON = ('Rewrite each block in new words. Reply only with JSON: {"blocks": [{"id": ..., "text": ...}]}.\n'
           "B1: Libraries give people a quiet place to read.\nB2: They also lend books for free.")
BLOCK = {"type": "object", "required": ["id", "text"],
         "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}
SCHEMA = {"type": "object", "required": ["blocks"], "properties": {"blocks": {"type": "array", "items": BLOCK}}}


def attempt(ask) -> tuple[bool, str]:
    """Run one request: whether it worked, and what to print."""
    started, usage = time.monotonic(), {}
    try:
        reply = ask(usage)
    except APIError as exc:
        return False, " ".join(str(exc).split())[:300]
    tokens = f"{usage.get('total_input_tokens', '?')} in, {usage.get('total_output_tokens', '?')} out"
    return True, f"{time.monotonic() - started:.1f} s, {tokens}: {reply}"


def check_model(base_url: str, key: str, model: str, system: str, thinking: str, temperature: float) -> bool:
    print(f"\n{model}")
    plain_ok, plain = attempt(lambda usage: chat(base_url, key, model, PLAIN, temperature, retries=2, system=system,
                                                 thinking=thinking, usage=usage, timeout=120))
    print(f"  text: {'ok' if plain_ok else 'FAILED'}, {plain[:220]}")
    json_ok, reply = attempt(lambda usage: chat(base_url, key, model, AS_JSON, temperature, retries=2, system=system,
                                                thinking=thinking, json_schema=SCHEMA, usage=usage, timeout=120))
    if json_ok:
        try:
            blocks = json.loads(reply.split(": ", 1)[1]).get("blocks", [])
            json_ok = {block.get("id") for block in blocks} == {"B1", "B2"}
        except (ValueError, AttributeError):
            json_ok = False
    print(f"  JSON: {'ok' if json_ok else 'FAILED'}, {reply[:220]}")
    return plain_ok and json_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("providers", nargs="+", choices=[name for name in PROVIDERS if name != "gemini"])
    parser.add_argument("--list", action="store_true", help="only list the models on offer (free ones first)")
    parser.add_argument("--models", help="comma-separated (default the provider's model and fallbacks)")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args(argv)
    load_env()
    system = style_instructions(load_style_profile(STYLE))
    print(f"Each request carries the built-in style profile ({len(system):,} characters).")
    failed = False
    for name in args.providers:
        provider = PROVIDERS[name]
        key = os.environ.get(provider.key_env, "")
        print(f"\n== {provider.label} ==")
        if args.list:
            try:
                print("\n".join(list_models(provider, key, timeout=30)))
            except APIError as exc:
                print(exc)
                failed = True
            continue
        if not key:
            print(f"No {provider.key_env} yet: add it to .env or paste it under Model & connection.")
            failed = True
            continue
        accepted = key_accepted(provider, key)
        print("Key: " + {True: "accepted", False: "REJECTED", None: "couldn't be checked just now"}[accepted])
        if accepted is False:
            failed = True
            continue
        models = args.models.split(",") if args.models else [provider.model, *provider.fallbacks]
        for model in filter(None, (model.strip() for model in models)):
            failed |= not check_model(provider.base_url, key, model, system, args.thinking, args.temperature)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
