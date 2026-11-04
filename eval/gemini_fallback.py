"""Last hosted resort for the generators: the Gemini website, through the unofficial gemini_webapi client.

It signs in with the session in the owner's Firefox, read fresh for each request (browser-cookie3), so nothing
needs updating by hand. Reading a browser's login is opt-in: GEMINI_WEB_FALLBACK=on in .env, and never in public
mode or the downloadable app. The website takes no temperature, system prompt or JSON format: the system prompt
goes in front of the prompt, and the JSON shape is asked for in words. The packages are imported only when used.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import threading
from pathlib import Path

WEB_MODEL = "gemini-web"
WEB_TIMEOUT = 120
# What the website sends as an ordinary reply when it fails. It must never be saved as a sample.
FAILED_REPLY = "i encountered an error doing what you asked"

ROOT = Path(__file__).resolve().parent.parent
_TURN = threading.Lock()  # one request at a time: this is one person's login
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


async def _ask(message: str, timeout: int) -> str:
    from gemini_webapi import GeminiClient, set_log_level

    set_log_level("ERROR")
    # Otherwise the client also opens every other browser's cookies on each sign-in, which is slow and can ask
    # for the Keychain. By module path, since gemini_webapi.utils.get_access_token can also name the function.
    importlib.import_module("gemini_webapi.utils.get_access_token").load_browser_cookies = lambda *a, **k: {}
    client = GeminiClient(*firefox_session())
    await client.init(timeout=timeout, auto_close=False, auto_refresh=False, watchdog_timeout=timeout)
    try:
        return (await client.generate_content(message)).text
    finally:
        await client.close()


def chat_web(prompt: str, system: str | None = None, json_schema: dict | None = None,
             timeout: int = WEB_TIMEOUT) -> str:
    """Ask the website once, on a client of its own, since batches call this from several threads. The whole
    request gets `timeout`, since the client's own retries could otherwise run for many minutes."""
    message = web_prompt(prompt, system, json_schema)
    with _TURN:
        try:
            text = asyncio.run(asyncio.wait_for(_ask(message, timeout), timeout))
        except GeminiWebError:
            raise
        except Exception as exc:  # the client raises its own errors, and ImportError when not installed
            # By type only: a sign-in error's text could include the request's cookies.
            raise GeminiWebError(f"the Gemini website failed ({type(exc).__name__})") from None
    text = (text or "").strip()
    if not text:
        raise GeminiWebError("empty response from the Gemini website")
    if text.lower().startswith(FAILED_REPLY):
        raise GeminiWebError("the Gemini website replied with its error message")
    fenced = _FENCE.match(text) if json_schema else None
    return fenced.group(1).strip() if fenced else text
