"""The app's two looks, picked under Appearance in the sidebar, each with a light and a dark mode:

- **Glass**: frosted, translucent panels floating over a soft aurora, with a ring dial for the score.
- **Instrument**: a forensic scanner read-out. A fine grid, hairline panels with corner brackets,
  monospace labels and a needle gauge. Built dark first; its light mode reads like a lab print-out.

Streamlit's own light/dark setting (Settings in the app menu) still picks the mode. The look only
layers styles on top, so every widget keeps working in either. The verdict tones (`human`, `review`,
`ai`) are the colour-blind-checked ones from `components.TONES` and stay the same in every look.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "ui" / "assets"
PREFS = ROOT / "data" / "ui_prefs.json"  # data/ is gitignored; local runs only

LOOKS = ("glass", "instrument")
LOOK_LABELS = {"glass": "Glass", "instrument": "Instrument"}
KEY = "keep:appearance:look"
DEFAULT_LOOK = os.environ.get("OPENDETECTOR_LOOK", "glass").strip().lower()
if DEFAULT_LOOK not in LOOKS:
    DEFAULT_LOOK = "glass"

# Each look's surface tokens per mode. Text pairs were checked at 4.5:1 or better against `bg` and `panel`.
TOKENS = {
    ("glass", "light"): {
        "bg": "#EEF0F8", "panel": "rgba(255,255,255,0.58)", "panel-solid": "#FFFFFF",
        "panel-strong": "rgba(255,255,255,0.78)",
        "edge": "rgba(255,255,255,0.85)", "hair": "rgba(20,24,48,0.08)", "ink": "#0E1222", "muted": "#545A72",
        "accent": "#5B57F5", "accent-2": "#0E9FB8", "glow": "rgba(91,87,245,0.28)", "track": "rgba(20,24,48,0.08)",
        "shadow": ("0 1px 0 rgba(255,255,255,0.9) inset, 0 18px 50px -18px rgba(40,44,90,0.28), "
                   "0 2px 6px rgba(40,44,90,0.06)"),
        "aurora": ("radial-gradient(60rem 40rem at 8% -10%, rgba(140,132,255,0.38), transparent 60%),"
                   "radial-gradient(50rem 36rem at 100% 0%, rgba(80,210,235,0.30), transparent 60%),"
                   "radial-gradient(56rem 40rem at 70% 110%, rgba(255,160,200,0.28), transparent 60%)"),
    },
    ("glass", "dark"): {
        "bg": "#06070D", "panel": "rgba(24,27,44,0.52)", "panel-solid": "#12141F",
        "panel-strong": "rgba(30,34,54,0.72)",
        "edge": "rgba(255,255,255,0.10)", "hair": "rgba(255,255,255,0.07)", "ink": "#ECEEF7", "muted": "#A0A6BE",
        "accent": "#8F8CFF", "accent-2": "#3FD3EA", "glow": "rgba(143,140,255,0.35)",
        "track": "rgba(255,255,255,0.08)",
        "shadow": ("0 1px 0 rgba(255,255,255,0.08) inset, 0 24px 60px -20px rgba(0,0,0,0.7), "
                   "0 2px 8px rgba(0,0,0,0.3)"),
        "aurora": ("radial-gradient(60rem 40rem at 5% -10%, rgba(92,72,255,0.30), transparent 60%),"
                   "radial-gradient(50rem 36rem at 100% 5%, rgba(20,170,200,0.20), transparent 60%),"
                   "radial-gradient(56rem 40rem at 65% 115%, rgba(170,60,220,0.20), transparent 60%)"),
    },
    ("instrument", "dark"): {
        "bg": "#05070A", "panel": "rgba(9,15,20,0.86)", "panel-solid": "#0A1117",
        "panel-strong": "rgba(12,20,27,0.94)",
        "edge": "rgba(90,230,255,0.20)", "hair": "rgba(90,230,255,0.10)", "ink": "#D9F4FA", "muted": "#7896A1",
        "accent": "#45E0FF", "accent-2": "#B6FF5C", "glow": "rgba(69,224,255,0.45)",
        "track": "rgba(90,230,255,0.10)",
        "shadow": "0 0 0 1px rgba(0,0,0,0.6), 0 20px 50px -24px rgba(0,0,0,0.9)",
        "aurora": ("linear-gradient(rgba(90,230,255,0.045) 1px, transparent 1px) 0 0 / 32px 32px,"
                   "linear-gradient(90deg, rgba(90,230,255,0.045) 1px, transparent 1px) 0 0 / 32px 32px,"
                   "radial-gradient(70rem 50rem at 50% -20%, rgba(69,224,255,0.10), transparent 70%)"),
    },
    ("instrument", "light"): {
        "bg": "#EEF1F0", "panel": "rgba(255,255,255,0.92)", "panel-solid": "#FFFFFF", "panel-strong": "#FFFFFF",
        "edge": "rgba(0,70,90,0.24)", "hair": "rgba(0,70,90,0.12)", "ink": "#0A1E25", "muted": "#4A6470",
        "accent": "#00788C", "accent-2": "#4C7A00", "glow": "rgba(0,120,140,0.22)", "track": "rgba(0,70,90,0.10)",
        "shadow": "0 1px 0 rgba(0,40,50,0.04), 0 12px 32px -20px rgba(0,40,50,0.35)",
        "aurora": ("linear-gradient(rgba(0,90,110,0.06) 1px, transparent 1px) 0 0 / 32px 32px,"
                   "linear-gradient(90deg, rgba(0,90,110,0.06) 1px, transparent 1px) 0 0 / 32px 32px"),
    },
}

FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&"
         "family=Geist+Mono:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap")


def _public() -> bool:
    value = os.environ.get("OPENDETECTOR_PUBLIC", "").strip().lower()
    if value:
        return value in ("1", "on", "true", "yes")
    return ROOT.parts[:3] == ("/", "mount", "src")


def _saved_look() -> str:
    if _public():
        return DEFAULT_LOOK
    try:
        look = json.loads(PREFS.read_text()).get("look")
    except (OSError, ValueError, AttributeError):
        return DEFAULT_LOOK
    return look if look in LOOKS else DEFAULT_LOOK


def _save_look() -> None:
    if _public():
        return
    try:
        PREFS.parent.mkdir(parents=True, exist_ok=True)
        PREFS.write_text(json.dumps({"look": st.session_state[KEY]}))
    except OSError:
        pass  # a remembered look is a convenience, never worth an error


def current_look() -> str:
    look = st.session_state.get(KEY)
    if look not in LOOKS:
        look = st.session_state[KEY] = _saved_look()
    return look


def mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


def tokens() -> dict[str, str]:
    return TOKENS[(current_look(), mode())]


def look_picker() -> None:
    """The sidebar's Appearance control. Light or dark stays with Streamlit's own Settings."""
    current_look()
    st.subheader("Appearance")
    st.segmented_control("Look", LOOKS, format_func=LOOK_LABELS.get, key=KEY, required=True,
                         label_visibility="collapsed", on_change=_save_look)
    st.caption("Light or dark: app menu > Settings." if current_look() == "glass" else
               "Built dark first. Light or dark: app menu > Settings.")


def logo_paths() -> tuple[str, str]:
    """The sidebar logo and the small mark for the active look and mode."""
    suffix = "" if current_look() == "glass" else "-instrument"
    tone = "-dark" if mode() == "dark" else ""
    return str(ASSETS / f"logo{suffix}{tone}.svg"), str(ASSETS / f"mark{suffix}.svg")


def hero(eyebrow: str, title: str, lede: str, chips: list[str]) -> None:
    """The page's opening: a small status line, the title, one line on what it does, and fact chips."""
    import html

    chip_html = "".join(f'<span class="od-chip">{html.escape(chip)}</span>' for chip in chips)
    st.html(f"""
<header class="od-hero">
  <div class="od-hero-eyebrow"><span class="od-pulse" aria-hidden="true"></span>{html.escape(eyebrow)}</div>
  <h1 class="od-hero-title">{html.escape(title)}</h1>
  <p class="od-hero-lede">{html.escape(lede)}</p>
  <div class="od-chips">{chip_html}</div>
</header>""")


# Styles that dress Streamlit's own elements. Selectors use data-testid hooks and the st-key-* classes
# that st.container(key=...) adds, which are the stable ones.
BASE_CSS = """
@import url('FONTS');
:root { color-scheme: MODE; }
.stApp { background: var(--od-aurora), var(--od-bg) !important; background-attachment: fixed !important; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button {
  font-family: 'Geist', 'Inter', -apple-system, system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding-top: 4.2rem; max-width: 1240px; }
[data-testid="stSidebar"] { background: var(--od-panel) !important; border-right: 1px solid var(--od-hair) !important;
  -webkit-backdrop-filter: blur(28px) saturate(170%); backdrop-filter: blur(28px) saturate(170%); }
[data-testid="stSidebar"] h3 { font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--od-muted) !important; padding-top: 1.1rem !important; }
[data-testid="stSidebarHeader"] { padding-top: 1.4rem; }
[data-testid="stTopNav"] { flex: 0 0 auto !important; width: fit-content !important; max-width: fit-content;
  background: var(--od-panel); border: 1px solid var(--od-edge); border-radius: 999px;
  padding: 3px; gap: 2px; box-shadow: var(--od-shadow);
  -webkit-backdrop-filter: blur(24px) saturate(170%); backdrop-filter: blur(24px) saturate(170%); }
[data-testid="stTopNavLink"] { border-radius: 999px !important; padding: 0.3rem 0.95rem !important; }
[data-testid="stTopNavLink"][aria-current="page"], [data-testid="stTopNavLink"].active {
  background: var(--od-accent) !important; }
[data-testid="stTopNavLink"][aria-current="page"] *, [data-testid="stTopNavLink"].active * { color: #fff !important; }

/* Panels: bordered containers and the app's own cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.stVerticalBlock[class*="st-key-od-panel"], .od-glass {
  background: var(--od-panel); border: 1px solid var(--od-edge) !important;
  border-radius: var(--od-radius) !important;
  box-shadow: var(--od-shadow); -webkit-backdrop-filter: blur(26px) saturate(170%);
  backdrop-filter: blur(26px) saturate(170%); }
.stVerticalBlock[class*="st-key-od-panel"] { padding: 1.25rem 1.35rem 1.1rem; }

/* Inputs */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
  background: var(--od-panel-strong) !important; border-color: var(--od-hair) !important;
  border-radius: 14px !important; }
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
  border-color: var(--od-accent) !important; box-shadow: 0 0 0 4px var(--od-glow) !important; }
[data-testid="stTextArea"] textarea { line-height: 1.65; font-size: 0.98rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--od-panel-strong) !important;
  border-radius: 16px !important;
  border: 1px dashed var(--od-edge) !important; }

/* Buttons */
[data-testid="stBaseButton-primary"] { background: var(--od-accent) !important; border: 0 !important;
  color: #fff !important;
  border-radius: 999px !important; padding: 0.5rem 1.35rem !important; font-weight: 600 !important;
  box-shadow: 0 8px 24px -8px var(--od-glow), 0 1px 0 rgba(255,255,255,0.25) inset; transition: transform .15s,
  box-shadow .15s; }
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px);
  box-shadow: 0 12px 30px -8px var(--od-glow); }
[data-testid="stBaseButton-secondary"] { background: var(--od-panel-strong) !important;
  border-color: var(--od-hair) !important;
  border-radius: 999px !important; }
[data-testid="stButtonGroup"] { background: var(--od-track); border-radius: 999px; padding: 3px; width: fit-content; }
[data-testid="stButtonGroup"] button { border: 0 !important; border-radius: 999px !important;
  background: transparent !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--od-panel-solid) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.14), 0 0 0 1px var(--od-hair); }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: var(--od-ink) !important; font-weight: 600; }
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type { background: var(--od-accent) !important;
  box-shadow: 0 0 12px var(--od-glow); }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] { gap: 0.3rem; border-bottom: 1px solid var(--od-hair); }
[data-testid="stTab"] { padding: 0.55rem 0.9rem !important; }
[data-testid="stTab"] p { color: var(--od-muted) !important; }
[data-testid="stTab"][aria-selected="true"] p { color: var(--od-ink) !important; font-weight: 600; }
[data-testid="stTab"] .react-aria-SelectionIndicator { background: var(--od-accent) !important;
  height: 2px !important;
  box-shadow: 0 0 12px var(--od-glow); }
/* Expanders, alerts, tables */
[data-testid="stExpander"] details { background: var(--od-panel); border: 1px solid var(--od-edge) !important;
  border-radius: 16px !important; }
[data-testid="stAlertContainer"] { border-radius: 16px !important; -webkit-backdrop-filter: blur(20px);
  backdrop-filter: blur(20px); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--od-hair); }

/* Hero */
.od-hero { margin: 0.4rem 0 1.4rem; }
.od-hero-eyebrow { display: inline-flex; align-items: center; gap: 0.55rem; font-size: 0.74rem; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--od-muted); }
.od-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--od-accent-2);
  box-shadow: 0 0 0 0 var(--od-accent-2); animation: od-pulse 2.4s ease-out infinite; }
@keyframes od-pulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--od-accent-2) 60%, transparent); }
  100% { box-shadow: 0 0 0 10px transparent; } }
.od-hero-title { font-size: clamp(2.2rem, 4.4vw, 3.6rem) !important; line-height: 1.02 !important;
  font-weight: 700 !important;
  letter-spacing: -0.035em; margin: 0.7rem 0 0.7rem !important; padding: 0 !important; color: var(--od-ink);
  background: linear-gradient(100deg, var(--od-ink) 30%, var(--od-accent) 75%, var(--od-accent-2));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.od-hero-lede { font-size: 1.06rem; color: var(--od-muted); max-width: 46rem; margin: 0 0 1rem; line-height: 1.55; }
.od-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.od-chip { font-size: 0.78rem; font-weight: 500; padding: 0.3rem 0.75rem; border-radius: 999px; color: var(--od-ink);
  background: var(--od-panel); border: 1px solid var(--od-edge); -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px); }
.od-section-title { display: flex; align-items: baseline; gap: 0.7rem; margin: 0.6rem 0 0.2rem; }
.od-section-title b { font-size: 1.35rem; font-weight: 650; letter-spacing: -0.02em; color: var(--od-ink); }
.od-section-title span { font-size: 0.8rem; color: var(--od-muted); }
@media (max-width: 640px) { [data-testid="stMainBlockContainer"] { padding-top: 3.6rem; } }
@media (prefers-reduced-motion: reduce) { .od-pulse, .od-scan { animation: none !important; } }
"""

INSTRUMENT_CSS = """
.stApp p, .stApp label, .stApp li, .stApp button { font-family: 'Geist', 'Inter', system-ui, sans-serif; }
[data-testid="stSidebar"] { -webkit-backdrop-filter: none; backdrop-filter: none; }
[data-testid="stSidebar"] h3, .od-hero-eyebrow, .od-eyebrow, .od-chip, .od-zones, .od-foot, .od-engine-value,
.od-stats b, .od-stats span, .od-section-title span, [data-testid="stTab"] p,
[data-testid="stCaptionContainer"] {
  font-family: 'JetBrains Mono', 'Geist Mono', ui-monospace, monospace !important; }
[data-testid="stTab"] p { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem !important; }
[data-testid="stTopNavLink"] span:not([data-testid="stIconMaterial"]) { font-family: 'JetBrains Mono',
  monospace !important;
  text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem; }
.od-stats span { font-size: 0.66rem; letter-spacing: 0.04em; text-transform: uppercase; }
.od-stats b { font-size: 1.15rem; }
:root { --od-radius: 4px; }
[data-testid="stTopNav"] { border-radius: 6px; }
[data-testid="stTopNavLink"] { border-radius: 4px !important; }
[data-testid="stTopNavLink"][aria-current="page"] *,
  [data-testid="stTopNavLink"].active * { color: var(--od-bg) !important; }
[data-testid="stBaseButton-primary"] { border-radius: 4px !important; color: var(--od-bg) !important;
  text-transform: uppercase;
  letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }
[data-testid="stBaseButton-primary"] * { color: var(--od-bg) !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stButtonGroup"], [data-testid="stButtonGroup"] button,
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div,
  [data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"] details, [data-testid="stAlertContainer"], [data-testid="stDataFrame"], .od-chip {
  border-radius: 4px !important; }
.od-chip { text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.7rem; }
.od-hero-title { font-weight: 600 !important; letter-spacing: -0.045em;
  background: none; -webkit-text-fill-color: var(--od-ink); color: var(--od-ink); }
.od-hero-title em { font-style: normal; color: var(--od-accent); -webkit-text-fill-color: var(--od-accent);
  text-shadow: 0 0 24px var(--od-glow); }
.od-pulse { border-radius: 1px; }
/* Corner brackets on every panel */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.stVerticalBlock[class*="st-key-od-panel"], .od-glass { position: relative;
  background-image: linear-gradient(var(--od-accent), var(--od-accent)), linear-gradient(var(--od-accent),
  var(--od-accent)),
    linear-gradient(var(--od-accent), var(--od-accent)), linear-gradient(var(--od-accent), var(--od-accent));
  background-size: 14px 1.5px, 1.5px 14px, 14px 1.5px, 1.5px 14px;
  background-position: top left, top left, bottom right, bottom right; background-repeat: no-repeat; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--od-accent) !important;
  box-shadow: 0 0 14px var(--od-glow); }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: var(--od-bg) !important; }
[data-testid="stButtonGroup"] button p { font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
  text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type > div {
  background: var(--od-bg) !important; }
"""


def css() -> str:
    look, current = current_look(), mode()
    values = dict(tokens())
    variables = "".join(f"--od-{name}: {value};" for name, value in values.items())
    radius = "22px" if look == "glass" else "4px"
    style = BASE_CSS.replace("FONTS", FONTS).replace("MODE", current)
    if look == "instrument":
        style += INSTRUMENT_CSS
    return f":root {{{variables} --od-radius: {radius};}}" + style
