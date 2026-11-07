"""The app's two looks, each with a light and a dark mode. Bistro is the default; the Look switch at the
foot of the sidebar changes to Glass:

- **Bistro**: the kitchen. Cream paper, espresso ink and paprika, JetBrains Mono for titles, labels, numbers
  and buttons and Geist for reading. Cloche, the critic (components.cloche), gives the verdict with his face,
  engine readings print as a bill, and a running batch is an order ticket. The labels stay plain; the kitchen
  flavour goes in subtitles, empty screens and loading lines (`flavour`), never near warnings or errors.
- **Glass**: frosted, translucent panels floating over a soft aurora, with a ring dial for the score.

Streamlit's own light/dark setting (Settings in the app menu) still picks the mode. The look only
layers styles on top, so every widget keeps working in either. The verdict tones (`human`, `review`,
`ai`) are the colour-blind-checked ones from `components.TONES` and stay the same in every look.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from . import prefs

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "ui" / "assets"

LOOKS = ("bistro", "glass")
LOOK_LABELS = {"bistro": "Bistro", "glass": "Glass"}
KEY = "keep:appearance:look"
# Bistro's finish: Textured (paper grain, kraft sidebar, cards with depth, an order-pad text box, enamel buttons)
# or Standard (flat).
FINISHES = ("textured", "standard")
FINISH_LABELS = {"textured": "Textured", "standard": "Standard"}
FINISH_KEY = "keep:appearance:finish"
# The Textured finish's wall: white ceramic kitchen tiles (the default) or the paper of the Standard finish, grained.
WALLS = ("tiles", "paper")
WALL_LABELS = {"tiles": "Tiles", "paper": "Paper"}
WALL_KEY = "keep:appearance:wall"
DEFAULT_LOOK = os.environ.get("OPENDETECTOR_LOOK", "bistro").strip().lower()
if DEFAULT_LOOK not in LOOKS:
    DEFAULT_LOOK = "bistro"
# The kitchen names under each page's link in the sidebar, and over its title, in Bistro.
PAGE_FLAVOUR = {"detect": "The critic's table", "red-team": "The kitchen", "guide": "The menu",
                "settings": "Behind the counter"}

# Each look's surface tokens per mode. Text pairs were checked at 4.5:1 or better against `bg` and `panel`.
TOKENS = {
    ("bistro", "light"): {
        "bg": "#F8F1E4", "panel": "#FFFCF6", "panel-solid": "#FFFCF6", "panel-strong": "#FBF6EC",
        "edge": "#E7DAC4", "hair": "#EFE3CF", "ink": "#2A1F17", "muted": "#6E5B4B", "accent": "#C2410C",
        "accent-2": "#3F7A3A", "glow": "rgba(194,65,12,0.22)", "track": "#F2E8D6", "shadow": "0 3px 0 #E7DAC4",
        "aurora": "linear-gradient(transparent, transparent)", "side": "#F2E8D6", "dash": "#E0CFB5",
        "accent-deep": "#9A3412", "press": "#8A2E08", "on": "#2A1F17", "on-ink": "#FFFCF6", "soft": "#FBEBDD",
        "faint": "#8C7867",
    },
    ("bistro", "dark"): {
        "bg": "#18120E", "panel": "#221A14", "panel-solid": "#221A14", "panel-strong": "#2A2019",
        "edge": "#3B2E24", "hair": "#33271F", "ink": "#F3E7D3", "muted": "#BBA58E", "accent": "#EA6A2E",
        "accent-2": "#6FB466", "glow": "rgba(234,106,46,0.25)", "track": "#2E231B", "shadow": "0 3px 0 #0E0A07",
        "aurora": "linear-gradient(transparent, transparent)", "side": "#1F1712", "dash": "#4A3B2F",
        "accent-deep": "#F08A55", "press": "#7A2A0A", "on": "#F3E7D3", "on-ink": "#18120E", "soft": "#3A2418",
        "faint": "#9C8670",
    },
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
}

FONTS = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&"
         "family=Geist+Mono:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap")


def public() -> bool:
    """Whether this is a public deploy, which remembers nothing between visits (ui/redteam.py says more)."""
    value = os.environ.get("OPENDETECTOR_PUBLIC", "").strip().lower()
    if value:
        return value in ("1", "on", "true", "yes")
    return ROOT.parts[:3] == ("/", "mount", "src")


def _saved_look() -> str:
    look = DEFAULT_LOOK if public() else prefs.load().get("look")
    return look if look in LOOKS else DEFAULT_LOOK


def _save_look() -> None:
    if not public():
        prefs.update(look=st.session_state[KEY])


def _save_finish() -> None:
    if not public():
        prefs.update(finish=st.session_state[FINISH_KEY])


def _save_wall() -> None:
    if not public():
        prefs.update(wall=st.session_state[WALL_KEY])


def wall() -> str:
    """The Textured finish's wall, tiles unless the viewer picked paper."""
    value = st.session_state.get(WALL_KEY)
    if value not in WALLS:
        saved = None if public() else prefs.load().get("wall")
        value = st.session_state[WALL_KEY] = saved if saved in WALLS else WALLS[0]
    return value


def tiled() -> bool:
    """Whether the page sits on the tile wall: Bistro, Textured, Tiles."""
    return bistro() and finish() == "textured" and wall() == "tiles"


def finish() -> str:
    """Bistro's finish, Textured unless the viewer picked Standard."""
    value = st.session_state.get(FINISH_KEY)
    if value not in FINISHES:
        saved = None if public() else prefs.load().get("finish")
        value = st.session_state[FINISH_KEY] = saved if saved in FINISHES else FINISHES[0]
    return value


def current_look() -> str:
    look = st.session_state.get(KEY)
    if look not in LOOKS:
        look = st.session_state[KEY] = _saved_look()
    return look


def bistro() -> bool:
    return current_look() == "bistro"


def flavour(plain: str, kitchen: str) -> str:
    """`kitchen` in Bistro, `plain` in Glass: for loading lines and empty screens, never for warnings or errors."""
    return kitchen if bistro() else plain


def mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


def tokens() -> dict[str, str]:
    return TOKENS[(current_look(), mode())]


def table_accent() -> str:
    """The bar colour for a table's progress column. Tables draw on a canvas in the base theme's colours, which Glass
    turns to its own with a hue turn (GLASS_CSS), so in Glass the bar starts as Bistro's paprika and turns violet."""
    return tokens()["accent"] if bistro() else TOKENS[("bistro", mode())]["accent"]


def heading(text: str, note: str = "") -> None:
    """A heading inside a page or panel, in the look's heading type, with an optional muted note after it."""
    import html

    extra = f"<span>{html.escape(note)}</span>" if note else ""
    st.html(f'<div class="od-section-title"><b>{html.escape(text)}</b>{extra}</div>')


def logo_paths() -> tuple[str, str]:
    """The sidebar logo and the small mark for the active look and mode."""
    suffix = "" if current_look() == "glass" else "-bistro"
    tone = "-dark" if mode() == "dark" else ""
    return str(ASSETS / f"logo{suffix}{tone}.svg"), str(ASSETS / f"mark{suffix}.svg")


def page_header(title: str, subtitle: str, status: str | None = None, ready: bool = True,
                page: str | None = None) -> None:
    """The top of a page: its title and one line on what it is for, and on the right one status pill
    (the detector build, the text model) when the page has one. In Bistro the page's kitchen name
    (`PAGE_FLAVOUR[page]`) sits over the title."""
    import html

    pill = (f'<span class="od-status{"" if ready else " off"}"><i aria-hidden="true"></i>{html.escape(status)}</span>'
            if status else "")
    eyebrow = (f'<span class="od-page-eyebrow">{html.escape(PAGE_FLAVOUR[page])}</span>'
               if page in PAGE_FLAVOUR and bistro() else "")
    st.html(f'<header class="od-page"><div>{eyebrow}<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
            f"{pill}</header>")


def group(title: str, step: str | None = None, kitchen: str | None = None):
    """A bordered panel that opens with a small title, for one group of related settings or results, and
    in Bistro its kitchen name ("the recipe card"). Returns the container, to fill with `with`."""
    import html

    box = st.container(border=True)
    number = f'<span class="od-step">{html.escape(step)}</span>' if step else ""
    extra = f'<em> · {html.escape(kitchen)}</em>' if kitchen and bistro() else ""
    box.html(f'<div class="od-group">{number}{html.escape(title)}{extra}</div>')
    return box


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
[data-testid="stMainBlockContainer"] { padding-top: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: var(--od-panel) !important; border-right: 1px solid var(--od-hair) !important;
  -webkit-backdrop-filter: blur(28px) saturate(170%); backdrop-filter: blur(28px) saturate(170%); }
[data-testid="stSidebar"] h3 { font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--od-muted) !important; padding-top: 1.1rem !important; }
[data-testid="stSidebarHeader"] { padding-top: 1.4rem; }
/* The sidebar is the app's navigation: the pages at the top, Settings at the foot. */
[data-testid="stSidebar"][data-collapsed="false"] { width: 236px !important; min-width: 236px !important;
  max-width: 236px !important; }
[data-testid="stSidebarNav"] { padding-top: 0.4rem; }
[data-testid="stSidebarNavLink"] { border-radius: 12px !important; padding: 0.5rem 0.8rem !important;
  margin: 0.1rem 0; }
[data-testid="stSidebarNavLink"]:hover { background: var(--od-track) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: color-mix(in srgb, var(--od-accent) 16%, transparent) !important;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--od-accent) 40%, transparent); }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--od-ink) !important; font-weight: 600; }
[data-testid="stSidebarNavSeparator"] { display: none; }

/* Page header and grouped panels */
.od-page { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 0.8rem 1.2rem;
  padding-bottom: 1rem; margin-bottom: 0.3rem; border-bottom: 1px solid var(--od-hair); }
.od-page h1 { font-size: 1.75rem !important; font-weight: 650 !important; letter-spacing: -0.02em;
  line-height: 1.15 !important; margin: 0 !important; padding: 0 !important; color: var(--od-ink); }
.od-page p { margin: 0.3rem 0 0; font-size: 0.93rem; color: var(--od-muted); }
.od-status { display: inline-flex; align-items: center; gap: 0.55rem; font-size: 0.82rem; font-weight: 500;
  padding: 0.4rem 0.85rem; border-radius: 999px; color: var(--od-ink); background: var(--od-panel);
  border: 1px solid var(--od-edge); white-space: nowrap; }
.od-status i { width: 8px; height: 8px; border-radius: 50%; background: var(--od-accent-2);
  box-shadow: 0 0 8px var(--od-accent-2); }
.od-status.off i { background: var(--od-muted); box-shadow: none; }
.od-group { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--od-muted); }
.od-step { color: var(--od-accent); margin-right: 0.6rem; font-variant-numeric: tabular-nums; }
.od-empty { border: 1px dashed var(--od-edge); border-radius: var(--od-radius); padding: 3rem 1.5rem;
  text-align: center; color: var(--od-muted); font-size: 0.95rem; line-height: 1.6; }
.od-empty b { display: block; color: var(--od-ink); font-size: 1.05rem; margin-bottom: 0.3rem; }

/* Each page link is its own pill. The nav keeps Streamlit's width: it measures that width to decide which
   links fit, so shrinking it to its contents folds every link into the "more" menu. */
[data-testid="stTopNavLink"] { border-radius: 999px !important; padding: 0.3rem 0.95rem !important;
  background: var(--od-panel) !important; border: 1px solid var(--od-edge); box-shadow: var(--od-shadow);
  -webkit-backdrop-filter: blur(24px) saturate(170%); backdrop-filter: blur(24px) saturate(170%); }
[data-testid="stTopNavLink"]:hover { background: var(--od-panel-strong) !important; }
[data-testid="stTopNavLink"][aria-current="page"], [data-testid="stTopNavLink"].active {
  background: var(--od-accent) !important; border-color: transparent; }
[data-testid="stTopNavLink"][aria-current="page"] *, [data-testid="stTopNavLink"].active * { color: #fff !important; }

/* Panels: bordered containers and the app's own cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.stVerticalBlock:has(> .stElementContainer > .stHtml > .od-group), .stVerticalBlock[class*="st-key-od-box"],
.stVerticalBlock[class*="st-key-od-panel"], .od-glass {
  background: var(--od-panel); border: 1px solid var(--od-edge) !important;
  border-radius: var(--od-radius) !important;
  box-shadow: var(--od-shadow); -webkit-backdrop-filter: blur(26px) saturate(170%);
  backdrop-filter: blur(26px) saturate(170%); }
.stVerticalBlock[class*="st-key-od-panel"] { padding: 1.25rem 1.35rem 1.1rem; }

/* Inputs */
[data-baseweb="textarea"],
[data-testid="stTextAreaRootElement"], [data-baseweb="input"], [data-baseweb="select"] > div {
  background: var(--od-panel-strong) !important; border-color: var(--od-hair) !important;
  border-radius: 14px !important; }
[data-baseweb="textarea"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within, [data-baseweb="input"]:focus-within {
  border-color: var(--od-accent) !important; box-shadow: 0 0 0 4px var(--od-glow) !important; }
[data-testid="stTextArea"] textarea { line-height: 1.65; font-size: 0.98rem; }
/* The box clips what's inside it to its own rounded corners, so the focus ring and the paper inside it agree */
[data-baseweb="textarea"], [data-testid="stTextAreaRootElement"], [data-baseweb="input"] { overflow: hidden; }
[data-baseweb="textarea"] textarea,
[data-testid="stTextAreaRootElement"] textarea, [data-baseweb="input"] input { border-radius: inherit; }
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
/* The track goes on the row of buttons, not the whole control, so a label sits above the pill and not inside it.
   A segmented control wider than its column wraps onto a second row instead of running under the next one. */
[data-testid="stButtonGroup"] > div:last-child { background: var(--od-track); border-radius: 1.4rem; padding: 3px;
  max-width: 100%; box-sizing: border-box; flex-wrap: wrap; }
[data-testid="stButtonGroup"] button { border: 0 !important; border-radius: 999px !important;
  background: transparent !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--od-panel-solid) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.14), 0 0 0 1px var(--od-hair); }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: var(--od-ink) !important; font-weight: 600; }
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type { background: var(--od-accent) !important;
  box-shadow: 0 0 12px var(--od-glow); }
/* Form buttons dress like the page's own; text-only buttons take the accent */
[data-testid="stBaseButton-primaryFormSubmit"] { background: var(--od-accent) !important; border: 0 !important;
  color: #fff !important; border-radius: 999px !important; font-weight: 600 !important; }
[data-testid="stBaseButton-secondaryFormSubmit"] { background: var(--od-panel-strong) !important;
  border-color: var(--od-hair) !important; border-radius: 999px !important; }
[data-testid="stBaseButton-tertiary"], [data-testid="stBaseButton-tertiary"] p { color: var(--od-accent) !important; }
[data-testid="stBaseButton-tertiary"]:hover:not(:disabled) { background: var(--od-track) !important; }
/* A disabled button looks it, whatever its kind */
.stApp button:disabled, .stApp button[disabled] { opacity: 0.45 !important; box-shadow: none !important;
  filter: saturate(0.35); cursor: not-allowed !important; transform: none !important; }
/* Every button presses down */
.stApp button { transition: transform .08s ease, box-shadow .08s ease, background .15s; }
.stApp button:active:not(:disabled) { transform: translateY(2px); }
a, [data-testid="stMarkdownContainer"] a { color: var(--od-accent); }
[data-testid="stPopoverBody"] { background: var(--od-panel-solid) !important;
  border: 1px solid var(--od-edge) !important;
  border-radius: 16px !important; }
[data-baseweb="tag"] { background: color-mix(in srgb, var(--od-accent) 14%, var(--od-panel-solid)) !important;
  color: var(--od-ink) !important; border: 1px solid color-mix(in srgb, var(--od-accent) 35%, transparent) !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--od-ink) !important; fill: var(--od-ink) !important; }

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
/* The tile wall's broken end, at the foot of the Settings page (wall_end) */
.od-wall-end { position: relative; margin: 3rem -4rem -6rem; height: 300px; overflow: hidden; }
.od-wall-end img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: top; }
.od-wall-end span { position: absolute; left: 4.5rem; bottom: 1.6rem; font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; color: var(--od-muted); opacity: 0.8; }
/* The stage's hidden spots in the sidebar (ui/kitchen.show) take no room */
[class*="st-key-od-stage-"] { display: none !important; }
"""

# Glass on top of BASE_CSS. The base theme (.streamlit/config.toml) is Bistro's, so Glass takes back its own type
# for headings and turns the widgets it can't recolour from the paprika of the theme to its violet.
GLASS_CSS = """
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
  font-family: 'Geist', 'Inter', -apple-system, system-ui, sans-serif !important; }
[data-testid="stSlider"] [data-baseweb="slider"], [data-testid="stProgress"], [data-testid="stSpinner"] svg,
[data-testid="stSpinner"] i, [data-testid="stStatusWidget"] svg,
[data-testid="stExpanderIconSpinner"] { filter: hue-rotate(221deg) saturate(1.25); }
[data-baseweb="textarea"] textarea, [data-baseweb="input"] input, [data-baseweb="input"] > div,
[data-baseweb="select"] > div > div { background: transparent !important; }
/* White boxes, and violet wherever the theme's paprika would show: hovers, focus, the open menu, tables */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"], [data-testid="stExpander"] details, [data-testid="stPopoverBody"],
[data-baseweb="popover"] ul, [data-testid="stTextAreaRootElement"], [data-testid="stTextInputRootElement"],
[data-testid="stTextInputRootElement"] > div,
[data-testid="stNumberInputContainer"] > div, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background-color: var(--od-panel-strong) !important; }
.stApp a, .stApp a:visited, [data-testid="stCaptionContainer"] a { color: var(--od-accent) !important; }
[data-testid="stCheckbox"] label > div:first-of-type > div { background-color: #FFFFFF !important; }
[data-baseweb="select"] > div:focus-within, [data-baseweb="select"] > div[aria-expanded="true"] {
  border-color: var(--od-accent) !important; box-shadow: 0 0 0 4px var(--od-glow) !important; }
[data-baseweb="popover"] li[aria-selected="true"], [data-baseweb="popover"] li:hover {
  background: color-mix(in srgb, var(--od-accent) 12%, transparent) !important; }
[data-testid^="stBaseButton-secondary"]:hover:not(:disabled), [data-testid="stPopoverButton"]:hover,
[data-testid="stDownloadButton"] button:hover:not(:disabled) { border-color: var(--od-accent) !important;
  color: var(--od-accent) !important; }
[data-testid^="stBaseButton-secondary"]:hover:not(:disabled) p, [data-testid="stPopoverButton"]:hover p {
  color: var(--od-accent) !important; }
.stApp button:focus-visible, .stApp [role="tab"]:focus-visible, .stApp summary:focus-visible {
  outline: 2px solid var(--od-accent) !important; box-shadow: 0 0 0 4px var(--od-glow) !important; }
[data-testid="stTab"]:hover p, [data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:hover svg { color: var(--od-accent) !important; fill: var(--od-accent); }
[data-testid="stCheckbox"] label:hover > div:first-of-type { border-color: var(--od-accent) !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--od-accent) !important; }
/* Tables take the base theme's cream; a quarter turn of hue makes them Glass's pale blue and their bars violet */
[data-testid="stDataFrame"] { filter: hue-rotate(221deg); }
"""

MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

# Bistro on top of BASE_CSS: flat paper, 1.5px crust borders with a pressed shadow, espresso segments, paprika for
# the one action on a page, and the kitchen-ticket mono for titles, labels, numbers and buttons.
BISTRO_CSS = """
.stApp h1, .stApp h2, .stApp h3, .od-page h1, .od-group, .od-status, .od-eyebrow, .od-stats b, .od-engine-value,
.od-hero-eyebrow, .od-chip, .od-zones, .od-foot, [data-testid="stTab"] p, [data-testid="stMetricValue"],
[data-testid="stMetricLabel"] p, [data-testid="stSidebarNavLink"] span, [data-testid="stSidebarNavLink"] p,
[data-testid="stExpander"] summary p,
[data-testid="stButtonGroup"] button p, [data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-secondary"] p { font-family: MONO !important; }
[data-testid="stSidebar"] { background: var(--od-side) !important; border-right: 1.5px solid var(--od-edge) !important;
  -webkit-backdrop-filter: none; backdrop-filter: none; }
[data-testid="stSidebarNavLink"] { flex-wrap: wrap; border: 1.5px solid transparent; border-radius: 14px !important;
  padding: 0.55rem 0.8rem !important; }
[data-testid="stSidebarNavLink"] > span:has([data-testid="stIconMaterial"]) { display: none; }
[data-testid="stSidebarNavLink"] span { font-size: 0.875rem; color: var(--od-ink) !important; }
[data-testid="stSidebarNavLink"]::after { flex-basis: 100%; font-family: 'Geist', system-ui, sans-serif;
  font-size: 0.75rem; color: var(--od-muted); }
[data-testid="stSidebarNavLink"][href$="/"]::after, [data-testid="stSidebarNavLink"][href$="/check"]::after {
  content: "DETECT"; }
[data-testid="stSidebarNavLink"][href$="/red-team"]::after { content: "RED_TEAM"; }
[data-testid="stSidebarNavLink"][href$="/guide"]::after { content: "GUIDE"; }
[data-testid="stSidebarNavLink"][href$="/settings"]::after { content: "SETTINGS"; }
[data-testid="stSidebarNavLink"]:hover { background: color-mix(in srgb, var(--od-edge) 45%, transparent) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--od-panel) !important;
  border-color: var(--od-edge); box-shadow: 0 2px 0 var(--od-edge) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { font-weight: 700; }
.st-key-od-look { margin-top: 1.4rem; padding: 0.75rem; border-radius: 14px; background: var(--od-panel);
  border: 1.5px solid var(--od-edge); gap: 0.5rem; }
.st-key-od-look [data-testid="stButtonGroup"], .st-key-od-look [data-testid="stButtonGroup"] > div:last-child {
  width: 100%; }
.st-key-od-look [data-testid="stButtonGroup"] button { flex: 1 1 0; padding: 4px 8px !important; min-width: 4.6rem; }
.st-key-od-look .stElementContainer, .st-key-od-look [data-testid="stButtonGroup"] div:has(> button) {
  width: 100% !important; }
.stApp div:has(> textarea), .stApp div:has(> input) { background: var(--od-panel-strong) !important; }

.od-page { border-bottom: 1.5px dashed var(--od-dash); padding-bottom: 1.25rem; }
.od-page-eyebrow { display: block; font-family: MONO; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--od-accent); margin-bottom: 0.35rem; }
.od-page h1 { font-size: 2.25rem !important; font-weight: 800 !important; letter-spacing: -0.04em; }
.od-page p { font-size: 0.95rem; }
.od-status { font-size: 0.75rem; border: 1.5px solid var(--od-edge); }
.od-status i { background: var(--od-accent-2); box-shadow: none; }
.od-group { font-weight: 700; letter-spacing: 0.14em; }
.od-group em { font-style: normal; font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--od-faint); }
.od-step { margin-right: 0.15rem; }
.od-step::after { content: " · "; color: var(--od-muted); }
.od-empty { border: 1.5px dashed var(--od-dash); background: color-mix(in srgb, var(--od-panel) 60%, transparent); }
.od-empty b { font-family: MONO; }

[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.stVerticalBlock:has(> .stElementContainer > .stHtml > .od-group), .stVerticalBlock[class*="st-key-od-box"],
.stVerticalBlock[class*="st-key-od-panel"], .od-glass {
  border: 1.5px solid var(--od-edge) !important; -webkit-backdrop-filter: none; backdrop-filter: none; }
[data-baseweb="textarea"],
[data-testid="stTextAreaRootElement"], [data-baseweb="input"], [data-baseweb="select"] > div {
  border: 1.5px solid var(--od-edge) !important; border-radius: 16px !important; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed var(--od-dash) !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { border-radius: 16px !important;
  font-weight: 800 !important; background: var(--od-accent) !important; color: #FFFCF6 !important;
  box-shadow: 0 4px 0 var(--od-press); min-height: 3rem; padding: 0.5rem 1.35rem !important; }
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] * { color: #FFFCF6 !important;
  font-weight: 800 !important; }
[data-testid="stBaseButton-primaryFormSubmit"] p { font-family: MONO !important; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--od-accent-deep) !important; box-shadow: 0 4px 0 var(--od-press); }
[data-testid="stBaseButton-primary"]:active:not(:disabled),
[data-testid="stBaseButton-primaryFormSubmit"]:active:not(:disabled) { transform: translateY(3px);
  box-shadow: 0 1px 0 var(--od-press) !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  border: 1.5px solid var(--od-edge) !important; border-radius: 10px !important; background: var(--od-panel) !important;
  box-shadow: 0 2px 0 var(--od-edge); }
[data-testid="stBaseButton-secondary"]:active:not(:disabled),
[data-testid="stBaseButton-secondaryFormSubmit"]:active:not(:disabled) { box-shadow: 0 0 0 var(--od-edge); }
[data-testid="stBaseButton-secondaryFormSubmit"] p, [data-testid="stBaseButton-tertiary"] p,
[data-testid="stPopoverButton"] p, [data-testid="stDownloadButton"] p, [data-testid="stPageLink"] p {
  font-family: MONO !important; font-size: 0.8rem !important; }
[data-testid="stBaseButton-tertiary"] { border-radius: 10px !important; }
[data-testid="stPopoverButton"] { border: 1.5px solid var(--od-edge) !important; border-radius: 10px !important;
  background: var(--od-panel) !important; }
[data-testid="stButtonGroup"] > div:last-child { background: var(--od-track); border-radius: 999px; }
/* One heading style everywhere: the kitchen-ticket mono */
.stApp h3 { font-size: 1.15rem !important; letter-spacing: -0.02em; }
.od-section-title b, [data-testid="stMetricValue"] div, [data-testid="stHeadingWithActionElements"] {
  font-family: MONO !important; }
.od-section-title b { font-size: 1.1rem; font-weight: 800; }
[data-testid="stMetric"] [data-testid="stMetricValue"] div { font-weight: 800; letter-spacing: -0.03em; }
[data-testid="stExpander"] details summary:hover, [data-testid="stExpander"] details[open] > summary {
  background: color-mix(in srgb, var(--od-edge) 30%, transparent) !important; }
/* The verdict needle shoots past its place and settles */
.od-band b { animation: od-needle 1.4s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes od-needle { from { left: 0%; } }
/* One centred column until there is a result: the setup column fills the row, and the result column, holding only
   the empty plate, stays out. When a result or the timer arrives, the setup slides left and the result slides in. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .od-stage-empty)
  > [data-testid="stColumn"]:first-child {
  flex: 0 0 100% !important; max-width: 100% !important; width: 100% !important;
  padding-inline: max(0px, calc((100% - 780px) / 2)); }
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .od-stage-empty)
  > [data-testid="stColumn"]:last-child {
  display: none !important; }
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .od-stage-empty) { gap: 0 !important; }
[data-testid="stHorizontalBlock"]:has(.st-key-od-setup) > [data-testid="stColumn"] {
  transition: flex-basis .7s cubic-bezier(.65,0,.35,1), padding .7s cubic-bezier(.65,0,.35,1); }
[data-testid="stHorizontalBlock"]:has(.st-key-od-setup) > [data-testid="stColumn"]:last-child {
  animation: od-slide-in .7s cubic-bezier(.65,0,.35,1) both; }
@keyframes od-slide-in { from { opacity: 0; transform: translateX(60px); } }
/* The Lab's order button stays in reach at the foot of the screen */
.st-key-od-order { position: sticky; bottom: 0; z-index: 3; padding: 0.6rem 0 0.9rem;
  background: linear-gradient(180deg, transparent, var(--od-bg) 40%); }
.st-key-od-results { gap: 0.5rem; }
.st-key-od-chef-picks:has(.stElementContainer) { padding: 10px 14px; border-radius: 12px;
  border: 1px dashed var(--od-dash);
  background: color-mix(in srgb, var(--od-panel-strong) 70%, transparent); gap: 0.35rem; }
@media (prefers-reduced-motion: reduce) { .od-band b, [data-testid="stColumn"] { animation: none !important;
  transition: none !important; } }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--od-on) !important; box-shadow: none; }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: var(--od-on-ink) !important; font-weight: 700; }
[data-testid="stButtonGroup"] button p { font-size: 0.78rem !important; }
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type { box-shadow: none; }
[data-testid="stSlider"] [role="slider"] { background: var(--od-accent) !important;
  box-shadow: 0 0 0 3px var(--od-panel) !important; }
[data-testid="stTab"] p { font-size: 0.82rem !important; }
[data-testid="stTab"] .react-aria-SelectionIndicator { height: 3px !important; box-shadow: none; }
[data-testid="stExpander"] details { border: 1.5px solid var(--od-edge) !important; }
[data-testid="stDataFrame"] { border: 1.5px solid var(--od-edge); }
.od-hero-title { font-family: MONO; font-weight: 800 !important; letter-spacing: -0.045em; background: none;
  -webkit-text-fill-color: var(--od-ink); color: var(--od-ink); }
.od-pulse { background: var(--od-accent); }
.od-verdict { border-width: 1.5px; box-shadow: 0 3px 0 var(--od-edge); -webkit-backdrop-filter: none;
  backdrop-filter: none; }
.od-verdict::before { display: none; }
"""
for page, name in (("DETECT", "detect"), ("RED_TEAM", "red-team"), ("GUIDE", "guide"), ("SETTINGS", "settings")):
    BISTRO_CSS = BISTRO_CSS.replace(f'"{page}"', f'"{PAGE_FLAVOUR[name]}"')
BISTRO_CSS = BISTRO_CSS.replace("MONO", MONO)

# Bistro's Textured finish, per mode: paper grain, a kraft-paper sidebar, cards with real depth, a lined order-pad
# text box, glossy enamel buttons, sunken switch tracks and the order ticket a little crooked.
TEXTURED_TOKENS = {
    "light": {"grain": "rgba(120,90,60,0.08)", "grain-2": "rgba(255,255,255,0.55)", "kraft": "#E3CFAB",
              "kraft-line": "rgba(120,85,45,0.07)", "kraft-dot": "rgba(90,60,30,0.10)", "kraft-edge": "#C9B08A",
              "card-top": "#FFFDF8", "card-bottom": "#FBF3E4", "card-edge": "#D9C7AA", "card-lip": "#E2D1B5",
              "card-shade": "rgba(60,40,20,0.45)", "shine": "#FFFFFF", "pad": "#FFFBF2",
              "pad-line": "rgba(194,65,12,0.13)", "pad-margin": "rgba(194,65,12,0.25)", "sunk": "#E6D6BC",
              "sunk-shade": "rgba(60,40,20,0.28)", "enamel-top": "#E8662A", "enamel-mid": "#C2410C",
              "enamel-bottom": "#A8370A", "enamel-edge": "#7A2706", "seg-top": "#4A3829", "seg-bottom": "#24190F",
              "bg": "#F3EADB"},
    "dark": {"grain": "rgba(255,230,200,0.035)", "grain-2": "rgba(0,0,0,0.35)", "kraft": "#2A1F16",
             "kraft-line": "rgba(255,220,180,0.035)", "kraft-dot": "rgba(0,0,0,0.25)", "kraft-edge": "#3B2E24",
             "card-top": "#2B2119", "card-bottom": "#221A14", "card-edge": "#3B2E24", "card-lip": "#120D09",
             "card-shade": "rgba(0,0,0,0.6)", "shine": "rgba(255,240,220,0.06)", "pad": "#1E1712",
             "pad-line": "rgba(234,106,46,0.14)", "pad-margin": "rgba(234,106,46,0.28)", "sunk": "#120D09",
             "sunk-shade": "rgba(0,0,0,0.55)", "enamel-top": "#F0844B", "enamel-mid": "#EA6A2E",
             "enamel-bottom": "#C9551F", "enamel-edge": "#7A2A0A", "seg-top": "#F3E7D3", "seg-bottom": "#D9C7AA",
             "bg": "#18120E"},
}
TEXTURED_CSS = """
.stApp { background-color: var(--od-tx-bg) !important;
  background-image: radial-gradient(var(--od-tx-grain) 1px, transparent 1.3px),
    radial-gradient(var(--od-tx-grain-2) 1px, transparent 1.3px) !important;
  background-size: 7px 7px, 11px 11px !important; background-position: 0 0, 3px 5px !important; }
[data-testid="stSidebar"] { background-color: var(--od-tx-kraft) !important;
  background-image: repeating-linear-gradient(95deg, var(--od-tx-kraft-line) 0 2px, transparent 2px 7px),
    radial-gradient(var(--od-tx-kraft-dot) 1px, transparent 1.4px) !important;
  background-size: auto, 5px 5px !important; border-right: 1px solid var(--od-tx-kraft-edge) !important;
  box-shadow: inset -6px 0 12px -8px rgba(60,40,20,0.35); }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(180deg, var(--od-tx-card-top), var(--od-tx-card-bottom)) !important;
  border: 1px solid var(--od-tx-kraft-edge) !important;
  box-shadow: inset 0 1px 0 var(--od-tx-shine), 0 3px 6px -2px rgba(60,40,20,0.35) !important;
  transform: rotate(-0.6deg); }
.st-key-od-look { background: linear-gradient(180deg, var(--od-tx-card-top), var(--od-tx-card-bottom));
  border: 1px solid var(--od-tx-kraft-edge); box-shadow: 0 3px 6px -2px rgba(60,40,20,0.3); }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.stVerticalBlock:has(> .stElementContainer > .stHtml > .od-group), .stVerticalBlock[class*="st-key-od-box"],
.stVerticalBlock[class*="st-key-od-panel"], .od-verdict, .od-bill, .od-hygiene, .od-text, .od-panel,
[data-testid="stExpander"] details, .od-empty {
  background: linear-gradient(180deg, var(--od-tx-card-top), var(--od-tx-card-bottom)) !important;
  border: 1px solid var(--od-tx-card-edge) !important;
  box-shadow: inset 0 1px 0 var(--od-tx-shine), 0 2px 0 var(--od-tx-card-lip),
    0 16px 26px -16px var(--od-tx-card-shade) !important; }
.od-bill { background:
  radial-gradient(circle at 8px -2px, var(--od-tx-bg) 6px, transparent 6.5px) 0 0 / 16px 12px repeat-x,
  linear-gradient(180deg, var(--od-tx-card-top), var(--od-tx-card-bottom)) !important; }
.od-empty { border-style: dashed !important; }
[data-testid="stTextArea"] textarea { background-color: var(--od-tx-pad) !important;
  background-image: repeating-linear-gradient(180deg, transparent 0 25px, var(--od-tx-pad-line) 25px 26px),
    linear-gradient(90deg, transparent 0 34px, var(--od-tx-pad-margin) 34px 35px, transparent 35px) !important;
  background-attachment: local !important; line-height: 26px !important; padding: 4px 18px 6px 48px !important;
  font-size: 0.97rem !important; color: var(--od-ink) !important; caret-color: var(--od-accent);
  font-feature-settings: "ss01", "cv11"; letter-spacing: 0.005em; }
/* The text sits on the pad's rules: 26-pixel lines, the first baseline just above the first rule */
[data-testid="stTextArea"] textarea::placeholder { color: var(--od-faint) !important; font-style: italic; opacity: 1; }
[data-testid="stTextArea"] textarea::selection { background: color-mix(in srgb, var(--od-accent) 22%, transparent); }
[data-baseweb="textarea"],
[data-testid="stTextAreaRootElement"] { box-shadow: inset 0 2px 6px rgba(60,40,20,0.18) !important;
  border-radius: 14px !important; }
[data-baseweb="textarea"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within {
  box-shadow: inset 0 2px 6px rgba(60,40,20,0.18), 0 0 0 3px var(--od-glow) !important;
  border-color: var(--od-accent) !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  border: 1px solid var(--od-tx-enamel-edge) !important;
  background: linear-gradient(180deg, var(--od-tx-enamel-top) 0%, var(--od-tx-enamel-mid) 55%,
    var(--od-tx-enamel-bottom) 100%) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.45), inset 0 -2px 0 rgba(0,0,0,0.12), 0 4px 0 var(--od-tx-enamel-edge),
    0 12px 18px -8px rgba(122,39,6,0.6) !important; text-shadow: 0 -1px 0 rgba(0,0,0,0.3); }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  filter: brightness(1.07); }
[data-testid="stBaseButton-primary"]:active:not(:disabled),
[data-testid="stBaseButton-primaryFormSubmit"]:active:not(:disabled) {
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 1px 0 var(--od-tx-enamel-edge) !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stPopoverButton"] {
  background: linear-gradient(180deg, var(--od-tx-card-top), var(--od-tx-card-bottom)) !important;
  border: 1px solid var(--od-tx-kraft-edge) !important;
  box-shadow: inset 0 1px 0 var(--od-tx-shine), 0 2px 0 var(--od-tx-kraft-edge) !important; }
[data-testid="stButtonGroup"] > div:last-child { background: var(--od-tx-sunk) !important;
  box-shadow: inset 0 2px 4px var(--od-tx-sunk-shade), 0 1px 0 var(--od-tx-shine); }
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: linear-gradient(180deg, var(--od-tx-seg-top), var(--od-tx-seg-bottom)) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 2px 3px rgba(42,31,23,0.4) !important; }
[data-testid="stCheckbox"] label > div:first-of-type { box-shadow: inset 0 2px 4px var(--od-tx-sunk-shade) !important; }
.od-ticket { transform: rotate(0.4deg);
  box-shadow: 0 2px 0 var(--od-tx-card-lip), 0 20px 30px -18px var(--od-tx-card-shade);
  background-image: radial-gradient(circle at 8px -2px, var(--od-tx-bg) 6px, transparent 6.5px),
    repeating-linear-gradient(180deg, transparent 0 31px, rgba(42,31,23,0.035) 31px 32px) !important;
  background-size: 16px 12px, auto !important; background-repeat: repeat-x, repeat !important; }
.st-key-od-order { background: linear-gradient(180deg, transparent, var(--od-tx-bg) 40%); }
"""


# Glazed white tiles 88 px square with 3 px of grout, each lit from the top left (a bevel of light on two edges, shade
# on the others, a soft dome of glaze), and a sheen across the whole wall. In dark mode, charcoal tiles.
TILES = {
    "light": {"grout": "#D6D1C8", "glaze": "#FFFFFF", "glaze-2": "#F5F3EF", "glaze-3": "#E9E6E0",
              "light": "rgba(255,255,255,0.95)", "shade": "rgba(110,98,82,0.16)", "sheen": "rgba(255,255,255,0.32)",
              "plaster": "#E3DACB", "ridge": "rgba(140,122,98,0.28)"},
    "dark": {"grout": "#100D0B", "glaze": "#34302B", "glaze-2": "#2A2723", "glaze-3": "#211E1B",
             "light": "rgba(255,255,255,0.08)", "shade": "rgba(0,0,0,0.38)", "sheen": "rgba(255,255,255,0.04)",
             "plaster": "#3A3128", "ridge": "rgba(0,0,0,0.35)"},
}
TILE = 88


def tiles_css(current: str) -> str:
    t, size = TILES[current], f"{TILE}px {TILE}px"
    return f"""
.stApp {{ background-color: {t['grout']} !important;
  background-image:
    linear-gradient(115deg, transparent 30%, {t['sheen']} 44%, transparent 58%),
    linear-gradient(90deg, {t['grout']} 0 3px, transparent 3px),
    linear-gradient(180deg, {t['grout']} 0 3px, transparent 3px),
    linear-gradient(90deg, transparent 3px, {t['light']} 3px 5px, transparent 5px calc(100% - 2px),
      {t['shade']} calc(100% - 2px)),
    linear-gradient(180deg, transparent 3px, {t['light']} 3px 5px, transparent 5px calc(100% - 2px),
      {t['shade']} calc(100% - 2px)),
    radial-gradient(120% 120% at 30% 25%, {t['glaze']} 0%, {t['glaze-2']} 55%, {t['glaze-3']} 100%) !important;
  background-size: 100% 100%, {size}, {size}, {size}, {size}, {size} !important;
  background-position: 0 0 !important; background-attachment: fixed !important; }}
[data-testid="stMainBlockContainer"] {{ position: relative; }}
[data-testid="stMainBlockContainer"]::before {{ content: ""; position: absolute; top: 0.35rem; right: 16rem;
  width: 320px; height: 64px; pointer-events: none; background: url("{vegetables_uri()}") no-repeat right top / contain;
  filter: drop-shadow(0 4px 3px rgba(40,30,20,0.25)); }}
@media (max-width: 640px) {{ [data-testid="stMainBlockContainer"]::before {{ width: 200px; }} }}
"""


def vegetables_uri() -> str:
    """Vegetables spilled along the top of the page on the tile wall: a tomato, basil leaves, a chilli, a clove of
    garlic, a cherry tomato and a few peppercorns, from the Lab's ingredients."""
    import base64
    import re

    from .kitchen import ICONS

    inner = {name: re.sub(r"^<svg[^>]*>|</svg>$", "", svg) for name, svg in ICONS.items()}
    placed = [("basil", 8, 20, -35, 1.2), ("tomato", 52, 10, 12, 1.45), ("basil", 98, 26, 50, 0.95),
              ("chili", 140, 14, -18, 1.35), ("garlic", 196, 18, 22, 1.05), ("tomato", 238, 30, -8, 0.8),
              ("basil", 262, 8, 120, 0.85)]
    parts = [f'<g transform="translate({x} {y}) rotate({r} 16 16) scale({k})">{inner[name]}</g>'
             for name, x, y, r, k in placed]
    parts += [f'<circle cx="{x}" cy="{y}" r="2.3" fill="#2A1F17" opacity="0.8"/>'
              for x, y in ((40, 52), (128, 58), (186, 50), (230, 60), (300, 44))]
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 70">{"".join(parts)}</svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def wall_end() -> None:
    """The foot of the Settings page on the tile wall: the tiles break off over bare plaster, combed with adhesive,
    and a few lie on the floor where they fell. The tiler went home early."""
    import base64
    import html
    import random

    t = TILES[mode()]
    rng = random.Random(11)
    width, height = 1400, 320
    edge = []
    for x in range(0, width, TILE):  # a column of tiles snapped off at its own depth, the break a small crack
        depth = rng.choice((26, 26, 70, 114, 114, 158))
        edge += [(x, depth + rng.uniform(-5, 5)), (x + TILE * rng.uniform(0.3, 0.45), depth + rng.uniform(-9, 9)),
                 (x + TILE * rng.uniform(0.55, 0.75), depth + rng.uniform(-9, 9)),
                 (x + TILE, depth + rng.uniform(-5, 5))]
    edge = [(0, edge[0][1]), *edge]
    plaster = f"M0 {height} L" + " L".join(f"{x:.0f} {y:.0f}" for x, y in edge) + f" L{width} {height} Z"
    rim = "M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in edge)
    ridges = "".join(f'<path d="M{x} {y} q60 -18 120 0 t120 0 t120 0" fill="none" stroke="{t["ridge"]}" '
                     f'stroke-width="3" stroke-linecap="round"/>'
                     for y in range(150, 260, 14) for x in range(-60, width, 360))
    # The tiles' broken edge: their thickness in grout colour, a light lip, and a few chips still stuck on.
    shards = (f'<path d="{rim}" fill="none" stroke="{t["grout"]}" stroke-width="7" stroke-linejoin="round"/>'
              f'<path d="{rim}" fill="none" stroke="{t["light"]}" stroke-width="1.5" transform="translate(0 -3)"/>'
              + "".join(f'<path d="M{x:.0f} {y + 3:.0f} l{rng.uniform(10, 22):.0f} {rng.uniform(5, 12):.0f} '
                        f'l{rng.uniform(-8, 2):.0f} {rng.uniform(8, 14):.0f} z" fill="{t["glaze-2"]}" '
                        f'stroke="{t["grout"]}" stroke-width="1.2"/>' for x, y in edge[2::7]))

    def fallen(x: float, y: float, angle: float, broken: bool = False) -> str:
        crack = (f'<path d="M14 6 L40 44 L32 60 L58 82" fill="none" stroke="{t["shade"]}" stroke-width="2.2"/>'
                 if broken else "")
        return (f'<g transform="translate({x} {y}) rotate({angle} 44 44)"><rect x="4" y="10" width="84" height="84" '
                f'rx="3" fill="rgba(0,0,0,0.18)"/><rect width="84" height="84" rx="3" fill="url(#glaze)" '
                f'stroke="{t["grout"]}" stroke-width="2"/>{crack}</g>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           'preserveAspectRatio="xMidYMin slice">'
           f'<defs><radialGradient id="glaze" cx="0.3" cy="0.25" r="1"><stop offset="0" stop-color="{t["glaze"]}"/>'
           f'<stop offset="0.55" stop-color="{t["glaze-2"]}"/><stop offset="1" stop-color="{t["glaze-3"]}"/>'
           f'</radialGradient></defs><path d="{plaster}" fill="{t["plaster"]}"/>{ridges}{shards}'
           f'{fallen(980, 210, -14, True)}{fallen(1090, 228, 23)}{fallen(1185, 200, 61)}'
           f'<rect x="0" y="{height - 14}" width="{width}" height="14" fill="rgba(0,0,0,0.10)"/></svg>')
    data = base64.b64encode(svg.encode()).decode()
    st.html(f'<div class="od-wall-end"><img src="data:image/svg+xml;base64,{data}" alt="" aria-hidden="true">'
            f'<span>{html.escape("Behind the counter, the tiler went home early.")}</span></div>')


def css() -> str:
    look, current = current_look(), mode()
    values = dict(tokens())
    variables = "".join(f"--od-{name}: {value};" for name, value in values.items())
    style = BASE_CSS.replace("FONTS", FONTS).replace("MODE", current)
    if look == "bistro":
        style += BISTRO_CSS
        if finish() == "textured":
            variables += "".join(f"--od-tx-{name}: {value};" for name, value in TEXTURED_TOKENS[current].items())
            style += TEXTURED_CSS
            if wall() == "tiles":
                style += tiles_css(current)
    else:
        style += GLASS_CSS
    return f":root {{{variables} --od-radius: 22px;}}" + style
