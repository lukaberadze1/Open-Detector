"""The app's pages, built by functions so a page can link to another one without importing it."""

from __future__ import annotations

import streamlit as st


def detect_page() -> st.Page:
    from . import detect

    return st.Page(detect.render, title="Detect", icon=":material/search:", url_path="detect", default=True)


def _red_team() -> None:
    from . import redteam  # loads the generation helpers only when the page is opened

    redteam.render()


def redteam_page() -> st.Page:
    return st.Page(_red_team, title="Red team", icon=":material/science:", url_path="red-team")
