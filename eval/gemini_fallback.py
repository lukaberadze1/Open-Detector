"""Last hosted resort for the generators: the Gemini website, through the unofficial gemini_webapi client.

It signs in with the session in the owner's Firefox, read fresh for each request (browser-cookie3), so nothing
needs updating by hand. Reading a browser's login is opt-in: GEMINI_WEB_FALLBACK=on in .env, and never in public
mode or the downloadable app.
The website takes no temperature, top-p, top-k, answer length or enforced JSON. Length and JSON are checked on
the reply instead, and a high temperature picks among the replies offered; top-p and top-k have no counterpart.
The rest does: the nearest model, thinking, and the system prompt as a gem. Packages are imported only when used.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

WEB_MODEL = "gemini-web"
WEB_TIMEOUT = 120
WEB_PARALLEL = 2  # requests at once: this is one person's login
HIGH_TEMPERATURE = 0.9
RETRY_WAITS = (5, 20)  # seconds before the second and third tries
# What the website sends as an ordinary reply when it fails. It must never be saved as a sample.
FAILED_REPLY = "i encountered an error doing what you asked"

ROOT = Path(__file__).resolve().parent.parent
_START = threading.Lock()
_LOOP = None
_CLIENT = None
_PLACES = None
_GEMS: dict = {}
_FENCE = re.compile(r"^```[a-z]*\s*\n(.*?)\n?```$", re.DOTALL)


class GeminiWebError(Exception):
    """The website couldn't answer. Never carries the cookies."""


def _public() -> bool:
    value = os.environ.get("OPENDETECTOR_PUBLIC", "").strip().lower()
    if value:
        return value in ("1", "on", "true", "yes")
    return ROOT.as_posix().startswith("/mount/src/")


def web_available() -> bool:
    """Whether this run may use the website: a private, non-packaged run that opted in."""
    if _public() or os.environ.get("OPENDETECTOR_PACKAGED", "").strip().lower() == "on":
        return False
    return os.environ.get("GEMINI_WEB_FALLBACK", "").strip().lower() == "on"


def firefox_session() -> tuple[str, str]:
    """The two Google session cookies from Firefox, and from no other browser."""
    import browser_cookie3

    found = {cookie.name: cookie.value for cookie in browser_cookie3.firefox(domain_name=".google.com")}
    if not found.get("__Secure-1PSID") or not found.get("__Secure-1PSIDTS"):
        raise GeminiWebError("no Google session in Firefox: sign in at gemini.google.com there")
    return found["__Secure-1PSID"], found["__Secure-1PSIDTS"]


def web_prompt(prompt: str, system: str | None = None, json_schema: dict | None = None) -> str:
    """The one message the website takes."""
    parts = [system] if system else []
    parts.append(prompt)
    if json_schema:
        parts.append("Reply with JSON only, no commentary and no code fence, matching this JSON schema:\n"
                     + json.dumps(json_schema))
    return "\n\n".join(parts)


def web_model(api_model: str | None) -> str | None:
    """The website's model nearest an API model name; None leaves the account's default."""
    name = (api_model or "").lower()
    if "gemini" not in name:
        return None
    return "gemini-flash-lite" if "lite" in name else "gemini-pro" if "pro" in name else "gemini-flash"


async def _gem_for(client, system: str):
    """A gem holding this system prompt, so it goes as a system prompt, as the API's does. One per distinct
    prompt, found again by name, so repeat requests don't add to the account's gems."""
    name = "open-detector-" + hashlib.sha256(system.encode()).hexdigest()[:12]
    if name not in _GEMS:
        await client.fetch_gems()
        found = client.gems.get(name=name)
        _GEMS[name] = found or await client.create_gem(name=name, prompt=system,
                                                       description="Open Detector red-team style profile")
    return _GEMS[name]


async def _client(timeout: int):
    """One signed-in client for the run, so a batch signs in once; a failure drops it (`_drop`), and the next
    request reads Firefox again."""
    global _CLIENT
    if _CLIENT is None:
        from gemini_webapi import GeminiClient, set_log_level

        set_log_level("ERROR")
        # Otherwise the client also opens every other browser's cookies on each sign-in, which is slow and can ask
        # for the Keychain. By module path, since gemini_webapi.utils.get_access_token can also name the function.
        importlib.import_module("gemini_webapi.utils.get_access_token").load_browser_cookies = lambda *a, **k: {}
        client = GeminiClient(*firefox_session())
        await client.init(timeout=timeout, auto_close=False, auto_refresh=False, watchdog_timeout=timeout)
        _CLIENT = client
    return _CLIENT


async def _drop() -> None:
    global _CLIENT
    client, _CLIENT = _CLIENT, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.close()


def pick_reply(candidates: list[str], temperature: float | None) -> str:
    """The first reply, as the website would show it, or at a high temperature any of the replies it offered.
    Not temperature itself, which the website doesn't take: only some of the variety a high one gives."""
    if temperature is not None and temperature >= HIGH_TEMPERATURE and len(candidates) > 1:
        return random.choice(candidates)
    return candidates[0] if candidates else ""


async def _ask(prompt: str, system: str | None, json_schema: dict | None, model: str | None, thinking: str | None,
               use_gem: bool, timeout: int, temperature: float | None = None) -> str:
    client = await _client(timeout)
    options = {"temporary": True, "extended_thinking": thinking == "high"}
    wanted = web_model(model)
    if wanted:
        with contextlib.suppress(ValueError):  # this account doesn't have it: its default model
            options["model"] = client.resolve_model(wanted)
    if system and use_gem:
        options["gem"] = await _gem_for(client, system)
        system = None
    response = await client.generate_content(web_prompt(prompt, system, json_schema), **options)
    return pick_reply([candidate.text for candidate in response.candidates], temperature) or response.text


async def _turn(ask, timeout: int) -> str:
    """One request in turn. Waiting for a place doesn't count against its time."""
    global _PLACES
    _PLACES = _PLACES or asyncio.Semaphore(WEB_PARALLEL)
    async with _PLACES:
        try:
            return await asyncio.wait_for(ask, timeout)
        except BaseException:
            await _drop()
            raise


def _run(ask, timeout: int) -> str:
    """Run a request on the one event loop the client lives on, from whichever thread a batch calls."""
    global _LOOP
    with _START:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(_turn(ask, timeout), _LOOP).result()


def _usable(text: str | None, json_schema: dict | None, max_tokens: int | None = None) -> str:
    """The reply, or a GeminiWebError when the website failed in the form of a reply. An answer past `max_tokens`
    (counted at 4 characters a token) fails as one cut off at the API's limit does."""
    text = (text or "").strip()
    if not text:
        raise GeminiWebError("empty response from the Gemini website")
    if text.lower().startswith(FAILED_REPLY):
        raise GeminiWebError("the Gemini website replied with its error message")
    if max_tokens and len(text) // 4 > max_tokens:
        raise GeminiWebError("the Gemini website's reply passed the length limit")
    if json_schema:
        fenced = _FENCE.match(text)
        text = fenced.group(1).strip() if fenced else text
        try:
            json.loads(text)
        except ValueError:
            raise GeminiWebError("the Gemini website's reply wasn't JSON") from None
    return text


def chat_web(prompt: str, system: str | None = None, json_schema: dict | None = None, timeout: int = WEB_TIMEOUT,
             *, model: str | None = None, thinking: str | None = None, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
    """Ask the website in a temporary chat. Each try gets `timeout`, since the client's own retries could otherwise
    run for many minutes, and tries again after RETRY_WAITS so one failure doesn't drop to Ollama. A try after the
    first sends the system prompt in the message, in case the gem was the problem."""
    failure = GeminiWebError("the Gemini website failed")
    for attempt, wait in enumerate((0, *RETRY_WAITS)):
        time.sleep(wait)
        try:
            ask = _ask(prompt, system, json_schema, model, thinking, attempt == 0, timeout, temperature)
            return _usable(_run(ask, timeout), json_schema, max_tokens)
        except GeminiWebError as exc:
            if "no Google session" in str(exc):
                raise
            failure = exc
        except Exception as exc:  # the client raises its own errors, and ImportError when not installed
            # By type only: a sign-in error's text could include the request's cookies.
            failure = GeminiWebError(f"the Gemini website failed ({type(exc).__name__})")
    raise failure
