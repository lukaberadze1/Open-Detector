"""Write AI "mirrors" of human essays: same assignment, similar length, different author.

Pairing each human essay with an AI essay on the same prompt means the detector is judged
on style, not topic. Defaults to Gemini through its Interactions API, with the key
read from GEMINI_API_KEY (or the project's .env). Any OpenAI-compatible endpoint works
(OpenAI, OpenRouter, a local LM Studio or Ollama server) via --base-url and --api-key-env,
so you can cover many model families. A local server needs no key and gets a 30-minute
timeout per request, since a small model on a laptop can take minutes per essay:

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 50 \\
        --base-url http://localhost:11434/v1 --model gemma3:4b --out data/essays/mirrors_gemma.jsonl

When the hosted models fail (a spent quota, a rejected key, an outage), the rest of the run
switches to a local Ollama model, gemma3:4b by default; --local-fallback '' turns that off.
Each essay's model column says which model wrote it.

Replies are unwrapped before they are saved: chat preambles ("Here's my essay!"), sign-offs
("Would you like me to...?"), --- rules and Markdown emphasis are removed.

    python -m eval.generate_essays --persuade persuade_2.0.csv --n 200 \\
        --style student --out data/essays/mirrors_gemini_student.jsonl

Styles: plain (default), student (a matched academic-writing condition), humanize (a detector-evasion
condition that asks the model to revise the generated sample). These are labelled benchmark conditions;
all generated samples remain labelled AI in the output data.

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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from opendetector.env import load_env

GEMINI_API_URL = "https://generativelanguage.googleapis.com"
GEMINI_OPENAI_URL = f"{GEMINI_API_URL}/v1beta/openai"
DEFAULT_MODEL = "gemini-3.8-flash"
# The free tier allows very few requests on the newest Flash model (20 when this was written);
# Flash-Lite's allowance is larger, so it is the fallback when the main model's quota runs out.
FALLBACK_MODEL = "gemini-3.5-flash-lite"
# The last resort once the hosted models fail: a model on a local Ollama server, with no key or quota.
LOCAL_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "gemma3:4b"

PROMPTS = {
    "plain": "Write an essay responding to the assignment below. Aim for about {words} words. "
             "Reply with the essay only.\n\nAssignment:\n{assignment}",
    "student": "You are a grade {grade} student. Write your essay for the assignment below in your own "
               "natural voice, about {words} words. Reply with the essay only.\n\nAssignment:\n{assignment}",
}
HUMANIZE = ("For this detector-evasion benchmark condition, revise the AI-generated sample to reduce the "
            "likelihood that an AI-text detector flags it. Preserve its claims and approximate length. "
            "This output is synthetic evaluation data and remains labelled AI. "
            "Reply with the revised sample only.\n\n{essay}")
STYLE_INTRO = ("Write in the style of the writer profiled below. Follow every rule exactly. The spelling slips, "
               "missing apostrophes, stray capitals and spacing quirks are part of the style: keep them, don't "
               "correct them. The reference section is the measured style the rules come from.")
# For a profile with registers: one register is sent, so its habits can't leak into another.
REGISTER_INTRO = ("Write in the style of the writer profiled below, in their {register} register. Every habit, "
                  "pattern and slip has a rate measured from their own text: use each about that often, scattered "
                  "unevenly, never clustered and never in every sentence. A habit whose status is 'once' was seen a "
                  "single time: use it at most once in a long text, or not at all. Keep the slips, missing "
                  "apostrophes, stray capitals and spacing quirks; don't correct them.")
REGISTER_NAMES = {"finalized": "finished work", "notes": "quick notes"}


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
        return [f"- {'; '.join(f'{k}: {v}' for k, v in x.items())}" if isinstance(x, dict)
                else f"- {json.dumps(x) if isinstance(x, list) else x}" for x in value]
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


def profile_registers(profile: dict) -> list[str]:
    """The registers a profile describes (finished work, quick notes), first the default; [] for a
    profile in the older single-register format."""
    registers = profile.get("registers")
    return list(registers) if isinstance(registers, dict) else []


def style_instructions(profile: dict, register: str | None = None) -> str:
    """The system prompt for a style profile. With registers: how to use the profile, the chosen
    register (the first when `register` is None or unknown) with its golden rule first, and what holds
    in every register; the other register is left out. Otherwise the overall rule, the hard rules,
    then the measurements as reference; profiles without system_prompt_overrides are all reference."""
    if registers := profile_registers(profile):
        name = register if register in registers else registers[0]
        chosen = dict(profile["registers"][name])
        golden = chosen.pop("golden_rule", None)
        lines = [REGISTER_INTRO.format(register=REGISTER_NAMES.get(name, name))]
        if golden:
            lines += ["", golden]
        if profile.get("how_to_use"):
            lines += ["", "# How to use this profile", profile["how_to_use"]]
        lines += ["", f"# The {REGISTER_NAMES.get(name, name)} register", *_render(chosen, 2)]
        if profile.get("shared"):
            lines += ["", "# In every register", *_render(profile["shared"], 2)]
        return "\n".join(lines)
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


_RULE = re.compile(r"^ {0,3}([-*_])(?: *\1){2,} *$")
_HEADING = re.compile(r"^ {0,3}#{1,6} +", re.M)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
# An opening that announces the essay, or one that ends in a colon.
_PREAMBLE = re.compile(r"\b(?:here(?:['\u2019]s| is| are)|below is)\b.{0,60}?\b(?:essay|letter|draft|version|response|"
                       r"rewrite)|:$", re.I)
_SIGN_OFF = re.compile(r"^\(?(?:(?:i )?hope (?:this|that|it) helps|let me know if you|would you like me|do you want me|"
                       r"want me to|feel free to (?:ask|let me)|is there anything else|note:|word count)", re.I)


def clean_essay(reply: str) -> str:
    """The essay inside a chat reply. Models wrap it ("Okay, here's my essay!", a --- rule, a Markdown
    title, "Would you like me to change the tone?"), and a detector trained on the wrapper learns to
    spot chat replies, not AI writing. Markdown emphasis goes too: an essay pasted into a form has none.
    A short opening paragraph is only dropped when it announces the essay, and a short closing one when
    it talks to the requester, so "Here is why..." and "I hope you agree." stay."""
    blocks, current = [], []
    for line in reply.replace("\r\n", "\n").split("\n"):
        if _RULE.match(line):
            blocks.append(current)
            current = []
        else:
            current.append(line)
    blocks.append(current)
    # A rule separates the essay from the chat around it, so the essay is the longest block.
    body = "\n".join(max(blocks, key=lambda b: len(" ".join(b).split())))
    body = _ITALIC.sub(r"\2", _BOLD.sub(r"\2", _HEADING.sub("", body)))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    def chatter(p: str, pattern: re.Pattern) -> bool:
        return len(p.split()) <= 40 and bool(pattern.search(p))

    if len(paragraphs) > 1 and chatter(paragraphs[0], _PREAMBLE):
        paragraphs.pop(0)
    while len(paragraphs) > 1 and chatter(paragraphs[-1], _SIGN_OFF):
        paragraphs.pop()
    return "\n\n".join(paragraphs) or reply.strip()


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_WAIT = 120  # seconds; a server asking us to wait longer than this means a spent quota
# Seconds per request. A small model on a small machine can take minutes per essay, and a request
# that times out starts over, so a local server gets far longer than a hosted API.
REMOTE_TIMEOUT, LOCAL_TIMEOUT = 300, 1800
_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?")
# How long a Gemini model reasons before it answers: less is faster and spends less quota.
THINKING_LEVELS = ("minimal", "low", "medium", "high")
# The levels each model accepts, from Google's thinking guide (October 2026). A model not listed here
# is asked for the level as given, and asked again without one if it turns that level down.
_THINKING_SUPPORT = {
    "gemini-3.6-flash": THINKING_LEVELS,
    "gemini-3.5-flash-lite": THINKING_LEVELS,
    "gemini-3-flash-preview": THINKING_LEVELS,
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
    "gemini-3-pro-preview": ("low", "high"),
    "gemini-2.5-pro": ("low", "medium", "high"),
    "gemini-2.5-flash": ("low", "medium", "high"),
    "gemini-2.5-flash-lite": ("low", "medium", "high"),
}
# A laptop runs one local model, so the groups of a batch written at the same time take turns with it:
# each request gets the whole machine, and its timeout doesn't run out while it waits in Ollama's queue.
_LOCAL_TURN = threading.Lock()


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


# Gemini models that turn a temperature down: they are asked at their own.
_NO_TEMPERATURE = frozenset({"gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash"})


def takes_temperature(model: str) -> bool:
    return model not in _NO_TEMPERATURE


def thinking_level(model: str, level: str | None) -> str | None:
    """The level to ask `model` for: `level` if the model has it, else the next one up it has (Gemini
    3.8 Flash starts at low), else its highest."""
    supported = _THINKING_SUPPORT.get(model)
    if level is None or supported is None or level in supported:
        return level
    return next((other for other in THINKING_LEVELS[THINKING_LEVELS.index(level):] if other in supported),
                supported[-1])


def chat(base_url: str, api_key: str, model: str, prompt: str, temperature: float, retries: int = 5,
         system: str | None = None, timeout: int = REMOTE_TIMEOUT, thinking: str | None = None,
         json_schema: dict | None = None, usage: dict | None = None) -> str:
    """Use Google's current Interactions API for Gemini, or another provider's OpenAI API. `thinking`
    is a Gemini thinking level, and `json_schema` makes Gemini reply with JSON in that shape; other
    providers get their own default thinking, and only the prompt asks them for JSON. Gemini's token
    counts for the request (total_input_tokens, total_output_tokens and so on) are added to `usage`."""
    if urllib.parse.urlparse(base_url).hostname == "generativelanguage.googleapis.com":
        return _gemini_interaction(model, api_key, prompt, temperature, retries, system, timeout,
                                   thinking_level(model, thinking), json_schema, usage)

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


def _gemini_interaction(model: str, api_key: str, prompt: str, temperature: float, retries: int,
                        system: str | None, timeout: int, thinking: str | None = None,
                        json_schema: dict | None = None, usage: dict | None = None) -> str:
    """Create a text interaction through Google's first-party Gemini API."""
    payload = {"model": model, "input": prompt}
    if system:
        payload["system_instruction"] = system
    if json_schema:
        payload["response_format"] = {"type": "text", "mime_type": "application/json", "schema": json_schema}
    config = {}
    if takes_temperature(model):
        config["temperature"] = temperature
    if thinking:
        config["thinking_level"] = thinking
    attempt = 0
    while attempt < retries:
        if config:
            payload["generation_config"] = config
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.load(response)
            if data.get("status") not in (None, "completed"):
                error = data.get("error")
                message = error.get("message") if isinstance(error, dict) else str(error or "")
                message = message or f"interaction status was {data.get('status')}"
                raise APIError(f"{model}: {message}")
            content = "".join(
                part.get("text", "")
                for step in data.get("steps", []) if step.get("type") == "model_output"
                for part in step.get("content", []) if part.get("type") == "text"
            ).strip()
            if not content:
                raise APIError(f"empty response from {model}")
            if usage is not None:
                usage.update(data.get("usage") or {})
            return content
        except urllib.error.HTTPError as exc:
            message, delay = _error_details(exc)
            lower = message.lower()
            # A model without this thinking level or JSON mode: ask again without it. The prompt still
            # asks for JSON, and the reply is checked either way.
            if exc.code == 400 and (("thinking" in lower and config.pop("thinking_level", None))
                                    or (("response_format" in lower or "schema" in lower)
                                        and payload.pop("response_format", None))):
                attempt -= 1
                payload.pop("generation_config", None)
                continue
            if exc.code == 429 and delay is not None and delay > MAX_WAIT:
                raise QuotaExhausted(f"{model} is out of quota: {message}") from exc
            if exc.code not in RETRYABLE_STATUS or attempt == retries:
                raise APIError(f"HTTP {exc.code} from Gemini Interactions API: {message}") from exc
            reason, wait = f"HTTP {exc.code}", delay if delay is not None else 2 ** (attempt - 1) * 5
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise APIError(f"could not reach Gemini Interactions API: {getattr(exc, 'reason', exc)}") from exc
            reason, wait = str(getattr(exc, "reason", exc)), 2 ** (attempt - 1) * 5
        print(f"  Gemini request failed ({reason}); retrying in {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
    raise AssertionError("unreachable")


def answer_token_limit(words: int) -> int:
    """The most tokens a local model may write for an answer of about `words` words: about twice what
    it needs, so a model that has lost track can't run on for hundreds of words."""
    return 2 * words + 200


def local_context_size(system: str | None, prompt: str, max_tokens: int) -> int:
    """A context window that holds the whole prompt and the answer. Ollama starts a model with a small
    one (4,096 tokens) and silently drops the start of a longer prompt, which is where the style
    profile and the instructions are. Counted at 3 characters a token (English runs nearer 4) and
    rounded up to a multiple of 8,192, so that the prompts of one batch share one size: a new size
    makes Ollama reload the model and lose what it has cached."""
    tokens = (len(system or "") + len(prompt)) // 3 + max_tokens + 256
    return -(-tokens // 8192) * 8192


def chat_local(local_url: str, model: str, prompt: str, temperature: float, system: str | None = None,
               max_tokens: int | None = None, context: int | None = None, timeout: int = LOCAL_TIMEOUT,
               json_schema: dict | None = None) -> str:
    """Ask a local server, one request at a time. Ollama gets its own chat endpoint, which, unlike its
    OpenAI-compatible one, takes a context window, an answer length and a JSON schema. Every block of a
    batch starts with the same document, so Ollama reads it once and keeps it cached. A server without
    that endpoint (LM Studio) gets the OpenAI-compatible request."""
    with _LOCAL_TURN:
        return _chat_local(local_url, model, prompt, temperature, system, max_tokens, context, timeout, json_schema)


def _chat_local(local_url: str, model: str, prompt: str, temperature: float, system: str | None,
                max_tokens: int | None, context: int | None, timeout: int, json_schema: dict | None) -> str:
    parsed = urllib.parse.urlparse(local_url)
    max_tokens = max_tokens or answer_token_limit(500)
    options = {"temperature": temperature, "num_predict": max_tokens,
               "num_ctx": context or local_context_size(system, prompt, max_tokens)}
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if json_schema:
        body["format"] = json_schema
    req = urllib.request.Request(
        f"{parsed.scheme}://{parsed.netloc}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        message, _ = _error_details(exc)
        if exc.code == 404 and "not found" not in message.lower():  # not Ollama: no /api/chat
            return chat(local_url, "", model, prompt, temperature, retries=1, system=system, timeout=timeout)
        raise APIError(f"HTTP {exc.code} from {local_url}: {message}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIError(f"could not reach {local_url}: {getattr(exc, 'reason', exc)}") from exc
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise APIError(f"empty response from {model}: {json.dumps(data)[:300]}")
    return content


def generate_with_fallback(base_url: str, api_key: str, models: list[str], prompt: str, temperature: float,
                           system: str | None, local_url: str = "", local_model: str = "",
                           max_tokens: int | None = None, local_context: int | None = None,
                           spent: set[str] | None = None, *, thinking: str | None = None,
                           json_schema: dict | None = None, timeout: int = 60) -> tuple[str, str, list[str]]:
    """Try the hosted models in order, advancing past quota and transient failures, then the local
    model, which needs no key and has no quota. `max_tokens` caps the local model's answer, and
    `local_context` fixes its context window (a batch passes one size for all its blocks). Models whose
    quota is spent for hours, not just busy for a moment, are added to `spent`. `thinking` and
    `json_schema` go to `chat`, and `timeout` is each hosted request's. Returns the text, the model
    that wrote it and the errors of the models that failed before it."""
    attempted = []
    errors = []
    for model_name in dict.fromkeys(name.strip() for name in models if name.strip()):
        attempted.append(model_name)
        try:
            result = chat(base_url, api_key, model_name, prompt, temperature, retries=1, system=system,
                          timeout=timeout, thinking=thinking, json_schema=json_schema)
            return result, model_name, errors
        except APIError as exc:
            errors.append(f"{model_name}: {exc}")
            if isinstance(exc, QuotaExhausted) and spent is not None:
                spent.add(model_name)
            message = str(exc)
            lower_message = message.lower()
            endpoint_unreachable = "could not reach" in lower_message and not any(
                word in lower_message for word in ("timed out", "timeout")
            )
            if any(status in message for status in ("HTTP 400", "HTTP 401", "HTTP 403")) or endpoint_unreachable:
                break
    if local_model:
        attempted.append(local_model)
        try:
            result = chat_local(local_url, local_model, prompt, temperature, system, max_tokens, local_context,
                                json_schema=json_schema)
            return result, local_model, errors
        except APIError as exc:
            errors.append(f"{local_model}: {exc}. Is Ollama running? Start it with `ollama serve` and "
                          f"fetch the model with `ollama pull {local_model}`.")
    details = "\n".join(f"- {error[:500]}" for error in errors) or "- No fallback models were configured."
    raise APIError(f"Every configured text model failed ({', '.join(attempted)}):\n{details}")


def unload_local_model(local_url: str, model: str, timeout: float = 30) -> bool:
    """Ask an Ollama server to drop `model` from memory now rather than after its five-minute
    keep-alive, so something else (the detector) has room. Other servers lack this endpoint;
    then nothing happens and the result is False."""
    parsed = urllib.parse.urlparse(local_url)
    req = urllib.request.Request(f"{parsed.scheme}://{parsed.netloc}/api/generate",
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (OSError, ValueError):  # URLError and HTTPError are OSErrors; a malformed URL is a ValueError
        return False


def local_model_problem(local_url: str, model: str, timeout: float = 2) -> str | None:
    """Why `model` can't be used on the local server right now, or None if it can. Asks Ollama for
    its downloaded models; a server without that endpoint (LM Studio) answered, so it is running,
    and is assumed to have the model."""
    parsed = urllib.parse.urlparse(local_url)
    server = f"{parsed.scheme}://{parsed.netloc}"
    try:
        with urllib.request.urlopen(f"{server}/api/tags", timeout=timeout) as response:
            downloaded = {entry.get("name") for entry in json.load(response).get("models", [])}
    except urllib.error.HTTPError:
        return None
    except (OSError, ValueError):
        return f"nothing is answering at {server}. Start Ollama with `ollama serve`."
    if (model if ":" in model else f"{model}:latest") not in downloaded:
        return f"{model} isn't downloaded. Fetch it with `ollama pull {model}`."
    return None


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
    parser.add_argument("--local-fallback", default=LOCAL_MODEL,
                        help="when the hosted models fail, switch to this model on --local-url ('' to just stop)")
    parser.add_argument("--local-url", default=LOCAL_URL)
    parser.add_argument("--style", choices=["plain", "student", "humanize"], default="plain")
    parser.add_argument("--style-profile", type=Path,
                        help="JSON style profile to write in, sent as the system prompt; its file name is "
                             "added to the attack label (<style>+<name>)")
    parser.add_argument("--register", help="with a profile that has registers, the one to write in "
                                           "(finalized or notes; default: the profile's first)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--base-url", default=GEMINI_API_URL)
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
            system = style_instructions(load_style_profile(args.style_profile), args.register)
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

    def write_essay(base_url: str, key: str, model: str, prompt: str) -> str:
        timeout = LOCAL_TIMEOUT if is_local(base_url) else REMOTE_TIMEOUT
        essay = clean_essay(chat(base_url, key, model, prompt, args.temperature, system=system, timeout=timeout))
        if args.style == "humanize":
            essay = clean_essay(chat(base_url, key, model, HUMANIZE.format(essay=essay), args.temperature,
                                     system=system, timeout=timeout))
        return essay

    base_url, key, model, written = args.base_url, api_key, args.model, 0
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
                    essay = write_essay(base_url, key, model, prompt)
                    break
                except QuotaExhausted as e:
                    if args.fallback_model and model != args.fallback_model:
                        print(f"{e}\nSwitching to {args.fallback_model} for the rest of the run.", file=sys.stderr)
                        model = args.fallback_model
                        continue
                    failure = e
                except APIError as e:
                    failure = e
                if not args.local_fallback or base_url == args.local_url:
                    hint = (f" Is the local server running? Start it with `ollama serve` and fetch the model with "
                            f"`ollama pull {model}`.") if base_url == args.local_url else ""
                    print(f"Stopped after {written} new essays: {failure}{hint}", file=sys.stderr)
                    return 1
                print(f"{failure}\nSwitching to {args.local_fallback} on {args.local_url} for the rest of the run.",
                      file=sys.stderr)
                base_url, key, model = args.local_url, "", args.local_fallback
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
