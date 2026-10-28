"""The Guide page: how to use the app, written for someone who has never set up anything like it, and
a place to save the Gemini API key the Red team page writes with. app.py opens it first the first time
the app runs on a computer. A public deploy leaves it out: visitors there paste their key in the sidebar."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

from eval.generate_essays import GEMINI_API_URL
from opendetector.config import DATA_ROOT, ROOT
from opendetector.env import load_env, save_env

from . import models, nav, style

KEY_NAME = "GEMINI_API_KEY"
KEY_PAGE = "https://aistudio.google.com/apikey"


def check_key(key: str) -> tuple[bool, str | None]:
    """Asks Google whether it accepts `key`: (whether to save it, what to tell the user if anything).
    A key Google turns down isn't saved; one that can't be checked for now is."""
    request = urllib.request.Request(f"{GEMINI_API_URL}/v1beta/models?pageSize=1", headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=15):
            return True, None
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403):
            return False, "Google didn't accept this key. Copy all of it again from the key page and paste it here."
        return True, f"Saved, but Google answered with an error ({exc.code}), so it couldn't be checked just now."
    except OSError:
        return True, "Saved, but this Mac couldn't reach Google to check it. Check your internet connection."


def _where() -> str:
    if DATA_ROOT == ROOT:
        return "the `.env` file in the project folder"
    shown = f"~/{DATA_ROOT.relative_to(Path.home())}" if DATA_ROOT.is_relative_to(Path.home()) else str(DATA_ROOT)
    return f"`{shown}`"


def _key_panel() -> None:
    load_env()
    saved = os.environ.get(KEY_NAME, "")
    with st.container(key="od-panel-guide-key"):
        st.subheader("Your Gemini key, for Red team only")
        st.markdown(
            "**Checking texts needs no key.** The Red team page writes AI samples with Google's Gemini, to test "
            "the detector, and for that it needs your own key. It's free:\n\n"
            f"1. Open [aistudio.google.com/apikey]({KEY_PAGE}) and sign in with a Google account.\n"
            "2. Press **Create API key**, then copy the key it shows.\n"
            "3. Paste it below and press **Save key**."
        )
        if saved:
            st.success(f"A key is saved, ending in …{saved[-4:]}. Red team will use it.", icon=":material/key:")
        with st.form("guide:key", clear_on_submit=True, border=False):
            key = st.text_input("Gemini API key", type="password", placeholder="Paste your key here",
                                label_visibility="collapsed")
            with st.container(horizontal=True):
                submitted = st.form_submit_button("Replace key" if saved else "Save key", type="primary",
                                                  icon=":material/key:")
        if submitted:
            key = key.strip()
            if not key:
                st.warning("Paste your key in the box first.")
            else:
                with st.spinner("Checking the key with Google…"):
                    keep, message = check_key(key)
                if keep:
                    save_env(KEY_NAME, key)
                    st.session_state["guide:notice"] = ("warning" if message else "success",
                                                        message or "Saved. Red team is ready to use.")
                    st.rerun()
                st.error(message, icon=":material/key_off:")
        if notice := st.session_state.pop("guide:notice", None):
            getattr(st, notice[0])(notice[1])
        if saved and st.button("Remove the key", icon=":material/delete:"):
            save_env(KEY_NAME, "")
            st.rerun()
        st.caption(f"The key is kept only on this computer, in {_where()}. Red team sends your instructions to "
                   "Google to write each sample; checking a text sends nothing anywhere.")


def render() -> None:
    style.hero("Welcome", "How Open Detector works",
               "Check whether a text was written by AI. The detector runs on this computer, so the texts you "
               "check never leave it.",
               ["Free and open source", "Works offline after the first check", "Word, PowerPoint, Markdown"])

    first, second = st.columns(2)
    with first, st.container(key="od-panel-guide-check"):
        st.subheader("1 · Check a text")
        st.markdown(
            "1. Open **Detect** at the top of the window.\n"
            "2. Paste at least 150 words, or choose **Upload a file** for a Word, PowerPoint or Markdown file.\n"
            "3. Press **Analyze**.\n\n"
            f"The very first check downloads the detector, about {models.download_gb(models.active_profile()):.1f} "
            "GB. That takes a few minutes on a slow connection, and it happens only once. After that, checks run "
            "without the internet in a few seconds."
        )
    with second, st.container(key="od-panel-guide-read"):
        st.subheader("2 · Read the result")
        st.markdown(
            "- **AI likelihood** runs from 0 to 100%, with a verdict: *Likely human*, *Needs review* or "
            "*Likely AI*.\n"
            "- Coloured sentences show where the AI-like writing is. Point at one to see its score.\n"
            "- Texts under 150 words, poems and lists get no verdict: the detector is built for prose.\n\n"
            "A score is evidence, not proof. Never use it as the only reason to accuse someone."
        )

    if nav.redteam_enabled():
        _key_panel()

    with st.container(key="od-panel-guide-more"):
        st.subheader("Good to know")
        st.markdown(
            "- **Look:** Glass or Instrument, under **Settings** at the bottom of the sidebar. For light or dark, "
            "use the **⋮** menu at the top right, then **Settings**.\n"
            "- **Slim or Full:** Slim, the strongest single engine, scores as well as Full on the project's tests "
            f"and needs about 1.5 GB of memory. Full adds two more engines, needs about 3.5 GB and downloads "
            f"about {models.download_gb('full') - models.download_gb('slim'):.1f} GB more.\n"
            "- **This guide** stays under **Guide** at the top of the window."
        )

    if st.button("Start checking texts", type="primary", icon=":material/arrow_forward:"):
        st.switch_page(nav.detect_page())
