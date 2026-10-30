"""A value kept in the visitor's own browser, for a public deploy that remembers nothing itself.

A public deploy keeps a visitor's Gemini key only in their session's memory, so it is gone when the tab
closes. A visitor who asks can keep it in their browser's storage for this site instead (localStorage):
it comes back on that device, in that browser, and nowhere else, and it is never written on the server,
which holds it in memory only to call Google, as it does for a key pasted by hand.

The browser side is a small Streamlit component, ui/assets/device_storage/index.html, which answers only
the app on its own site.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_storage = components.declare_component("device_storage", path=str(Path(__file__).parent / "assets/device_storage"))


def device_value(name: str, action: str = "read", value: str = "") -> str | None:
    """What this browser keeps under `name` once it has done `action`: "read", "store" (`value`) or
    "forget". None until the browser first answers; each change of its answer reruns the page."""
    return _storage(name=name, action=action, value=value if action == "store" else "",
                    key=f"device_storage:{name}", default=None)
