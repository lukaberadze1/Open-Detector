"""Open Detector's Streamlit app: streamlit run app.py

Two pages share one set of loaded models and the detector choice in the sidebar:
- Detect (ui/detect.py) scores pasted text, or a Markdown, Word or PowerPoint file.
- Red team (ui/redteam.py) generates labelled AI samples to test the detector with.

OPENDETECTOR_PROFILE=slim pins the slim build (Desklib alone on the CPU) for small hosts, and
OPENDETECTOR_REDTEAM=off leaves out the Red team page. OPENDETECTOR_PUBLIC=on runs the Red team
page for strangers: visitors bring their own key and see only their own work. The Streamlit Cloud
deploy sets the profile and public mode; the Hugging Face one leaves the Red team page out.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from ui import models, nav, style
from ui.components import apply_theme

ASSETS = Path(__file__).resolve().parent / "ui" / "assets"

st.set_page_config(page_title="Open Detector", page_icon=str(ASSETS / "mark.svg"), layout="wide")
logo, mark = style.logo_paths()
st.logo(logo, icon_image=mark, size="large")
apply_theme()

# Streamlit forgets the state of widgets the current page doesn't draw. Writing the "keep:" keys back
# on every run keeps each page's inputs while the user is on the other one.
for key in [key for key in st.session_state if isinstance(key, str) and key.startswith("keep:")]:
    st.session_state[key] = st.session_state[key]

pages = [nav.detect_page()]
if os.environ.get("OPENDETECTOR_REDTEAM", "on").strip().lower() not in ("0", "off", "false", "no"):
    pages.append(nav.redteam_page())
page = st.navigation(pages, position="top" if len(pages) > 1 else "hidden")
with st.sidebar:
    models.profile_picker()
page.run()
with st.sidebar:
    style.look_picker()
