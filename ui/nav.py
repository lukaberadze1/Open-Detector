"""The app's pages, built by functions so a page can link to another one without importing it."""

from __future__ import annotations

import os

import streamlit as st


def detect_page() -> st.Page:
    from . import detect

    return st.Page(detect.render, title="Detect", icon=":material/search:", url_path="detect", default=True)


def redteam_enabled() -> bool:
    """OPENDETECTOR_REDTEAM=off leaves the Adversarial Lab page out."""
    return os.environ.get("OPENDETECTOR_REDTEAM", "on").strip().lower() not in ("0", "off", "false", "no")


def _red_team() -> None:
    from . import redteam  # loads the generation helpers only when the page is opened

    redteam.render()


def redteam_page() -> st.Page:
    return st.Page(_red_team, title="Adversarial Lab", icon=":material/science:", url_path="red-team")


def guide_page() -> st.Page:
    from . import guide

    return st.Page(guide.render, title="Guide", icon=":material/help:", url_path="guide")
