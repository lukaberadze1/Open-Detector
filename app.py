"""Open Detector's Streamlit app: streamlit run app.py

Two pages share one set of loaded models and the detector build picked on the Detect page:
- Detect (ui/detect.py) scores pasted text, or a Markdown, Word or PowerPoint file.
- Adversarial Lab (ui/redteam.py) generates labelled AI challenge samples to test the detector.
A third, the Guide (ui/guide.py), explains both and saves the Gemini key the lab uses. It opens first
the first time the app runs on a computer; a public deploy leaves it out.

OPENDETECTOR_PROFILE=slim pins the slim build (Desklib alone on the CPU) for small hosts, and
OPENDETECTOR_REDTEAM=off leaves out the Adversarial Lab page. OPENDETECTOR_PUBLIC=on runs the lab
page for strangers: visitors bring their own key and see only their own work. The Streamlit Cloud
deploy sets the profile and public mode; the Hugging Face one leaves the lab page out.
"""

from __future__ import annotations

import streamlit as st

from ui import kitchen, nav, prefs, style
from ui.components import apply_theme

logo, mark = style.logo_paths()
st.set_page_config(page_title="Open Detector", page_icon=mark, layout="wide")
st.logo(logo, icon_image=mark, size="large")
apply_theme()

# Streamlit forgets the state of widgets the current page doesn't draw. Writing the "keep:" keys back
# on every run keeps each page's inputs while the user is on the other one.
# The detector build ("profile") is drawn on the Detect page only, so it is kept the same way.
for key in [key for key in st.session_state if isinstance(key, str) and (key.startswith("keep:") or key == "profile")]:
    st.session_state[key] = st.session_state[key]

pages = [nav.detect_page()]
if nav.redteam_enabled():
    pages.append(nav.redteam_page())
pages.append(nav.guide_page())
pages.append(nav.settings_page())
pages.append(nav.detect_alias())  # hidden: the old /detect address
page = st.navigation(pages, position="sidebar")
if not style.public() and not prefs.load().get("welcomed"):
    prefs.update(welcomed=True)  # the Guide opens first once per computer, then stays a tab away
    st.switch_page(nav.guide_page())
page.run()
# Cloche, in the corner of every page in Bistro, gets the run's latest cue (ui/kitchen.py).
kitchen.show({"red-team": "lab", "settings": "settings"}.get(page.url_path, "detect"))
