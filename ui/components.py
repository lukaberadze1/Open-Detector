"""Pieces both pages use: the theme's extra styles, verdict display, sentence highlights, charts, and
widgets that keep their state when the user switches pages."""

from __future__ import annotations

import base64
import hashlib
import html
import math
from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from opendetector.segment import count_words

# Label, Streamlit color name and icon for each verdict.
VERDICTS = {
    "likely_ai": ("Likely AI", "red", ":material/smart_toy:"),
    "needs_review": ("Needs review", "orange", ":material/flag:"),
    "likely_human": ("Likely human", "green", ":material/person:"),
    "insufficient_text": ("Too short", "gray", ":material/short_text:"),
    "not_prose": ("Not prose", "gray", ":material/format_list_bulleted:"),
}
# The three scored verdicts, lowest band first, and the tone that fills each one.
BANDS = {"likely_human": "human", "needs_review": "review", "likely_ai": "ai"}
# Tones for the light and dark palettes (.streamlit/config.toml). red, orange, green and gray are safe
# for text. human, review and ai fill highlights, gauges and chart marks; they were checked to stay
# apart under colour blindness, and in dark mode review and ai also need their underline styles.
TONES = {
    "light": {"red": "#DC2626", "orange": "#B45309", "green": "#15803D", "gray": "#64748B", "surface": "#FFFFFF",
              "ink": "#171A26", "muted": "#6B7083", "track": "#E9EBF2", "accent": "#4F46E5",
              "human": "#0F8A6E", "review": "#E0A100", "ai": "#D03B3B"},
    "dark": {"red": "#F87171", "orange": "#FBBF24", "green": "#4ADE80", "gray": "#94A3B8", "surface": "#0D0F16",
             "ink": "#E7E9F0", "muted": "#9298AB", "track": "#252937", "accent": "#8B93FF",
             "human": "#199E70", "review": "#C98500", "ai": "#E5484D"},
}

# Bistro's verdict colours (basil, mustard, tomato), on top of TONES. Review and AI keep their own underline styles
# (dotted, solid), so the three stay apart without colour. Dark mode keeps TONES' checked ones.
BISTRO_TONES = {"light": {"red": "#B42318", "orange": "#9A5B00", "green": "#3F7A3A", "human": "#3F7A3A",
                          "review": "#E9B949", "ai": "#B42318"}}

CSS = """
.od-verdict { position: relative; overflow: clip; border-radius: var(--od-radius, 22px);
  padding: 1.5rem 1.6rem 1.25rem;
  background: var(--od-panel); border: 1px solid var(--od-edge); box-shadow: var(--od-shadow);
  -webkit-backdrop-filter: blur(26px) saturate(170%); backdrop-filter: blur(26px) saturate(170%); }
.od-verdict::before { content: ""; position: absolute; inset: -40% -10% auto auto; width: 70%; height: 140%;
  background: radial-gradient(closest-side, color-mix(in srgb, var(--tone) 22%, transparent), transparent);
  pointer-events: none; }
.od-top { position: relative; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 1.2rem; }
.od-top > div { flex: 1 1 15rem; min-width: 0; }
.od-top > .od-dial { flex: 0 0 auto; margin: 0 auto; }
.od-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--od-muted); }
.od-label { font-size: clamp(1.9rem, 3vw, 2.6rem); font-weight: 700; letter-spacing: -0.03em; line-height: 1.05;
  color: var(--tone); margin: 0.35rem 0 0.5rem; }
.od-meaning { font-size: 0.95rem; color: var(--od-ink); opacity: 0.86; max-width: 30rem; line-height: 1.5; }
.od-dial { width: 230px; height: 196px; display: block; max-width: 100%; }
.od-priorities { width: 100%; max-width: 190px; height: auto; display: block; margin: 0 auto; }
.od-zones { position: relative; display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem; font-size: 0.78rem; }
.od-zones span { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.65rem 0.22rem 0.5rem;
  border-radius: 999px; border: 1px solid var(--od-hair); color: var(--od-muted); }
.od-zones span.on { color: var(--od-ink); border-color: color-mix(in srgb, var(--tone) 55%, transparent);
  background: color-mix(in srgb, var(--tone) 12%, transparent); font-weight: 600; }
.od-zones i { width: 8px; height: 8px; border-radius: 50%; }
.od-foot { position: relative; font-size: 0.78rem; color: var(--od-muted); margin-top: 0.9rem; padding-top: 0.8rem;
  border-top: 1px solid var(--od-hair); font-variant-numeric: tabular-nums; }

.od-panel { height: 100%; box-sizing: border-box; border-radius: var(--od-radius, 22px); padding: 1.3rem 1.4rem;
  background: var(--od-panel); border: 1px solid var(--od-edge); box-shadow: var(--od-shadow);
  -webkit-backdrop-filter: blur(26px) saturate(170%); backdrop-filter: blur(26px) saturate(170%); }
.od-engine { margin-top: 1rem; }
.od-engine-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; }
.od-engine-name { font-weight: 600; font-size: 0.95rem; color: var(--od-ink); }
.od-engine-value { font-weight: 600; font-variant-numeric: tabular-nums; color: var(--od-ink); }
.od-meter { position: relative; height: 6px; border-radius: 999px; background: var(--od-track);
  margin: 0.45rem 0 0.35rem; }
.od-meter i { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--od-accent-2), var(--od-accent)); box-shadow: 0 0 12px var(--od-glow); }
.od-note { font-size: 0.78rem; color: var(--od-muted); line-height: 1.4; }
.od-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(5rem, 1fr)); gap: 0.6rem;
  margin-top: 1.1rem; }
.od-stats div { padding: 0.65rem 0.75rem; border-radius: calc(var(--od-radius, 22px) * 0.6);
  background: var(--od-track); }
.od-stats b { display: block; font-size: 1.3rem; font-weight: 650; line-height: 1.2; color: var(--od-ink);
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.od-stats span { font-size: 0.72rem; color: var(--od-muted); }

.od-text { line-height: 1.9; font-size: 1.02rem; max-height: 36rem; overflow-y: auto; padding: 1.3rem 1.5rem;
  background: var(--od-panel); border: 1px solid var(--od-edge); border-radius: var(--od-radius, 22px);
  box-shadow: var(--od-shadow); color: var(--od-ink);
  -webkit-backdrop-filter: blur(26px) saturate(170%); backdrop-filter: blur(26px) saturate(170%); }
.od-sample { white-space: pre-wrap; max-height: 22rem; font-size: 0.95rem; line-height: 1.7; }
.od-hl { border-radius: 4px; padding: 1px 1px; -webkit-box-decoration-break: clone; box-decoration-break: clone;
  text-decoration-thickness: 2px; text-underline-offset: 4px; transition: box-shadow 0.15s; }
.od-hl.human { background: color-mix(in srgb, var(--od-human) var(--a), transparent); }
.od-hl.review { background: color-mix(in srgb, var(--od-review) var(--a), transparent);
  text-decoration-line: underline; text-decoration-style: dotted; text-decoration-color: var(--od-review); }
.od-hl.ai { background: color-mix(in srgb, var(--od-ai) var(--a), transparent);
  text-decoration-line: underline; text-decoration-style: solid; text-decoration-color: var(--od-ai); }
.od-hl:hover { box-shadow: 0 0 0 1.5px color-mix(in srgb, currentColor 45%, transparent); }
.od-hl.od-flash { animation: od-flash 1.6s ease-out; }
.od-pickable .od-hl { cursor: pointer; }
.od-hl.od-picked { box-shadow: 0 0 0 2px var(--od-accent); background-color: color-mix(in srgb, var(--od-accent) 16%,
  transparent); }
.od-cloche { flex: 0 0 auto; display: block; max-width: 100%; height: auto; }
.od-critic { position: relative; display: flex; flex-wrap: wrap; align-items: center; gap: 1.4rem; }
.od-critic > div { flex: 1 1 17rem; min-width: 0; }
.od-says { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 0.9rem; margin: 0.35rem 0 0.45rem; }
.od-says .od-label { font-family: var(--od-mono); font-size: clamp(1.7rem, 2.6vw, 2.15rem); font-weight: 800;
  letter-spacing: -0.03em; margin: 0; }
.od-says span { font-family: var(--od-mono); font-size: 1.1rem; color: var(--od-muted); }
.od-band { position: relative; display: flex; gap: 4px; height: 12px; margin-top: 0.9rem; }
.od-band i { border-radius: 2px; }
.od-band i:first-child { border-radius: 6px 2px 2px 6px; }
.od-band i:last-of-type { border-radius: 2px 6px 6px 2px; flex: 1 1 auto; }
.od-band i.human { background: color-mix(in srgb, var(--od-human) 26%, var(--od-panel)); }
.od-band i.review { background: color-mix(in srgb, var(--od-review) 45%, var(--od-panel)); }
.od-band i.ai { background: color-mix(in srgb, var(--od-ai) 28%, var(--od-panel)); }
.od-band b { position: absolute; top: -6px; width: 4px; height: 24px; margin-left: -2px; border-radius: 2px;
  background: var(--od-ink); }
.od-band-ends { display: flex; justify-content: space-between; gap: 0.5rem; margin-top: 0.4rem;
  font-family: var(--od-mono); font-size: 0.69rem; color: var(--od-muted); }
.od-bill { height: 100%; box-sizing: border-box; padding: 1.25rem 1.4rem; border-radius: 6px 6px 20px 20px;
  background: var(--od-panel) radial-gradient(circle at 8px -2px, var(--od-bg) 6px, transparent 6.5px) 0 0 / 16px 12px
  repeat-x; border: 1.5px solid var(--od-edge); font-family: var(--od-mono); font-size: 0.82rem; color: var(--od-ink);
  display: flex; flex-direction: column; gap: 0.55rem; }
.od-bill .od-eyebrow { margin-top: 0.25rem; }
.od-line { display: flex; align-items: baseline; gap: 0.5rem; }
.od-line i { flex: 1 1 auto; border-bottom: 2px dotted var(--od-dash); transform: translateY(-3px); }
.od-line b { font-weight: 700; }
.od-line.total { font-size: 0.95rem; font-weight: 800; }
.od-line.total i { border: 0; }
.od-rule { border-top: 1.5px dashed var(--od-dash); margin: 0.25rem 0; }
.od-bill small { font-size: 0.69rem; color: var(--od-muted); }
.od-hygiene { height: 100%; box-sizing: border-box; padding: 1.25rem; border-radius: 20px; background: var(--od-panel);
  border: 1.5px solid var(--od-edge); display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.82rem;
  line-height: 1.5; }
.od-billrow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.9rem; }
.od-billrow > .od-bill { flex: 1.6 1 17rem; min-width: 0; height: auto; }
.od-billrow > .od-hygiene { flex: 1 1 12rem; min-width: 0; height: auto; }
.od-ticket { padding: 1.4rem 1.5rem; border-radius: 6px 6px 22px 22px; border: 1.5px solid var(--od-edge);
  box-shadow: 0 3px 0 var(--od-edge);
  background: var(--od-panel) radial-gradient(circle at 8px -2px, var(--od-bg) 6px, transparent 6.5px) 0 0 / 16px 12px
  repeat-x; display: flex; flex-direction: column; gap: 0.85rem; color: var(--od-ink); }
.od-ticket-head { display: flex; align-items: center; gap: 1rem; }
.od-ticket-head > div { display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
.od-ticket-head .od-eyebrow { color: var(--od-accent); }
.od-ticket-head b { font-family: var(--od-mono); font-size: 1.2rem; font-weight: 800; overflow-wrap: anywhere; }
.od-ticket-head span { font-size: 0.82rem; color: var(--od-muted); }
.od-ticket ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
.od-ticket li { display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 10px; padding: 8px 10px;
  border-radius: 12px; }
.od-ticket li.now { background: var(--od-soft); }
.od-ticket li > i { font-style: normal; font-family: var(--od-mono); font-size: 0.875rem; font-weight: 800;
  line-height: 20px; color: var(--od-dash); }
.od-ticket li.done > i { color: var(--od-accent-2); }
.od-ticket li.now > i { color: var(--od-accent); }
.od-ticket li b { display: block; font-family: var(--od-mono); font-size: 0.875rem; line-height: 20px;
  font-weight: 600; }
.od-ticket li.now b { font-weight: 800; }
.od-ticket li.todo b { color: var(--od-muted); }
.od-ticket li span { display: block; font-size: 0.75rem; color: var(--od-muted); }
.od-ticket-bar { height: 8px; border-radius: 8px; background: var(--od-track); overflow: hidden; }
.od-ticket-bar i { display: block; height: 100%; border-radius: 8px; background: var(--od-accent); }
.od-ticket small { font-family: var(--od-mono); font-size: 0.75rem; color: var(--od-muted); }
@keyframes od-flash { 0%, 35% { box-shadow: 0 0 0 3px var(--od-accent); } 100% { box-shadow: 0 0 0 0 transparent; } }

.od-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1.2rem; font-size: 0.86rem;
  margin-bottom: 0.75rem; }
.od-key { display: inline-flex; align-items: center; gap: 0.45rem; }
.od-key i { display: inline-block; width: 1.5rem; height: 0.85rem; border-radius: 3px; box-sizing: border-box; }
.od-key i.human { background: color-mix(in srgb, var(--od-human) 22%, transparent); }
.od-key i.review { background: color-mix(in srgb, var(--od-review) 38%, transparent);
  border-bottom: 2px dotted var(--od-review); }
.od-key i.ai { background: color-mix(in srgb, var(--od-ai) 40%, transparent); border-bottom: 2px solid var(--od-ai); }
.od-key b { font-weight: 600; font-variant-numeric: tabular-nums; }
.od-hint { font-size: 0.8rem; opacity: 0.6; }
.od-map { display: flex; gap: 3px; height: 14px; padding: 3px; border-radius: 999px; background: var(--od-track); }
.od-run { min-width: 3px; border-radius: 999px; cursor: pointer; transition: transform 0.12s; }
.od-run.human { background: var(--od-human); }
.od-run.review { background: var(--od-review); }
.od-run.ai { background: var(--od-ai); }
.od-run.plain { background: var(--od-track); }
.od-run:hover { transform: scaleY(1.6); }
.od-ends { display: flex; justify-content: space-between; font-size: 0.72rem; opacity: 0.6; margin: 0.3rem 0 0.9rem; }
.od-tip { position: fixed; z-index: 1000; pointer-events: none; max-width: 20rem; padding: 0.45rem 0.7rem;
  border-radius: 0.7rem; background: var(--od-ink); color: var(--od-bg); font-size: 0.8rem; line-height: 1.4;
  white-space: pre-line; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2); }
.od-tip[hidden] { display: none; }
"""

# Wires up one highlights block: a tooltip that follows the pointer over anything with data-tip, and
# map runs that scroll to their first sentence and flash it. The block's id replaces ROOT.
HIGHLIGHT_JS = """
(() => {
  const root = document.getElementById("ROOT");
  if (!root || root.dataset.wired) return;
  root.dataset.wired = "1";
  const tip = root.querySelector(".od-tip");
  root.addEventListener("mousemove", (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) { tip.hidden = true; return; }
    if (tip.textContent !== target.dataset.tip) tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const below = event.clientY + 18 + tip.offsetHeight < window.innerHeight;
    tip.style.left = Math.max(8, Math.min(event.clientX + 12, window.innerWidth - tip.offsetWidth - 8)) + "px";
    tip.style.top = (below ? event.clientY + 18 : event.clientY - tip.offsetHeight - 12) + "px";
  });
  root.addEventListener("mouseleave", () => { tip.hidden = true; });
  // Scrolling moves the text out from under a still pointer, so the tip would point at nothing.
  window.addEventListener("scroll", () => { tip.hidden = true; }, { capture: true, passive: true });
  root.addEventListener("click", (event) => {
    const run = event.target.closest("[data-jump]");
    const sentence = run && document.getElementById(run.dataset.jump);
    if (!sentence) return;
    sentence.scrollIntoView({ block: "center", behavior: "smooth" });
    sentence.classList.remove("od-flash");
    void sentence.offsetWidth;
    sentence.classList.add("od-flash");
  });
})();
"""


def theme_type() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


def tones() -> dict[str, str]:
    """The verdict tones for the active mode and look: TONES, with Bistro's own in its light mode."""
    from . import style

    merged = dict(TONES[theme_type()])
    if style.bistro():
        merged.update(BISTRO_TONES.get(theme_type(), {}))
    return merged


def look_tones() -> dict[str, str]:
    """The verdict tones for the active mode, with the active look's surfaces (ui/style.py) on top."""
    from . import style

    merged = tones()
    look = style.tokens()
    merged.update({"surface": look["panel-solid"], "ink": look["ink"], "muted": look["muted"], "track": look["track"],
                   "accent": look["accent"]})
    return merged


def apply_theme() -> None:
    """Styles for the whole app: the active look (ui/style.py), then the app's own HTML (verdict card,
    highlights) in the verdict tones."""
    from . import kitchen, style

    variables = "".join(f"--od-{name}: {value};" for name, value in tones().items()
                        if name not in ("ink", "muted", "track", "accent", "surface"))
    st.html(f"<style>{style.css()}:root {{{variables}--od-surface: var(--od-panel-solid); --od-mono: {style.MONO};}}"
            f"{CSS}{kitchen.CSS}</style>")


# Cloche, the critic: a serving dome with a moustache and a monocle, from the Bistro identity. His face is the
# verdict, so he has a job: pleased for likely human, monocle up for needs review, shocked for likely AI, and
# steam while the text is being tasted. The colours are the look's (`CLOCHE_TONES`).
MOODS = {"likely_human": "happy", "needs_review": "skeptical", "likely_ai": "shocked"}
CLOCHE_LABELS = {"happy": "Cloche, pleased: likely human", "skeptical": "Cloche, raising a monocle: needs review",
                 "shocked": "Cloche, shocked: likely AI", "tasting": "Cloche, tasting"}
CLOCHE_TONES = {
    "bistro": {"ink": "#2A1F17", "dome": "#F3E7D3", "plate": "#FFFCF6", "shine": "#FFFBF3", "accent": "#C2410C",
               "blush": "rgba(194,65,12,0.20)", "sweat": "#9ED3F0"},
    "glass": {"ink": "#1E2A4A", "dome": "rgba(255,255,255,0.75)", "plate": "rgba(255,255,255,0.55)",
              "shine": "#FFFFFF", "accent": "#5B57F5", "blush": "rgba(244,114,182,0.35)", "sweat": "#A5D8FF"},
}


def cloche_svg(mood: str = "skeptical", tone: str = "bistro") -> str:
    """Cloche as SVG markup, 120 units square (the logo files are drawn from this too)."""
    c = CLOCHE_TONES.get(tone, CLOCHE_TONES["bistro"])
    ink, line = c["ink"], 'fill="none" stroke-linecap="round"'
    parts = []
    if mood == "tasting":
        parts += [f'<path d="M{x} {y} q-4 -5 0 -10 q4 -5 0 -10" {line} stroke="{c["accent"]}" stroke-width="2.6"/>'
                  for x, y in ((50, 26), (60, 22), (70, 26))]
    parts += [f'<ellipse cx="60" cy="95" rx="54" ry="8" fill="{c["plate"]}" stroke="{ink}" stroke-width="3"/>',
              f'<path d="M13 92 A47 47 0 0 1 107 92 Z" fill="{c["dome"]}" stroke="{ink}" stroke-width="3" '
              'stroke-linejoin="round"/>',
              f'<path d="M24 78 A37 37 0 0 1 44 52" {line} stroke="{c["shine"]}" stroke-width="4"/>',
              f'<circle cx="60" cy="40" r="6.5" fill="{c["accent"]}" stroke="{ink}" stroke-width="3"/>',
              f'<circle cx="34" cy="80" r="5" fill="{c["blush"]}"/><circle cx="86" cy="80" r="5" fill="{c["blush"]}"/>']
    if mood == "happy":
        parts += [f'<path d="{d}" {line} stroke="{ink}" stroke-width="{w}"/>' for d, w in
                  (("M38 59 q8 -5 16 0", 3), ("M66 59 q8 -5 16 0", 3), ("M40 70 q6 -7 12 0", 3.2),
                   ("M68 70 q6 -7 12 0", 3.2))]
    elif mood == "shocked":
        parts += [f'<path d="M37 56 l16 -5" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<path d="M67 51 l16 5" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<circle cx="46" cy="67" r="6.5" fill="#FFFFFF" stroke="{ink}" stroke-width="2.6"/>',
                  f'<circle cx="74" cy="67" r="6.5" fill="#FFFFFF" stroke="{ink}" stroke-width="2.6"/>',
                  f'<circle cx="46" cy="68" r="2.6" fill="{ink}"/><circle cx="74" cy="68" r="2.6" fill="{ink}"/>',
                  f'<ellipse cx="60" cy="87" rx="3.4" ry="2.6" fill="{ink}"/>',
                  f'<path d="M98 50 q4 6 0 9 q-4 -3 0 -9 z" fill="{c["sweat"]}" stroke="{ink}" stroke-width="1.6"/>']
    elif mood == "tasting":
        parts += [f'<path d="M38 58 q8 -3 16 0" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<path d="M66 58 q8 -3 16 0" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<circle cx="47" cy="65" r="3.6" fill="{ink}"/><circle cx="75" cy="65" r="3.6" fill="{ink}"/>']
    else:  # skeptical: one eyebrow up, the monocle and its chain
        parts += [f'<path d="M38 61 l15 1" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<path d="M65 54 q9 -7 18 -2" {line} stroke="{ink}" stroke-width="3"/>',
                  f'<circle cx="46" cy="69" r="3.6" fill="{ink}"/><circle cx="74" cy="68" r="3.6" fill="{ink}"/>',
                  f'<circle cx="74" cy="68" r="9.5" fill="none" stroke="{ink}" stroke-width="2.6"/>',
                  f'<path d="M83 71 q7 9 1 20" {line} stroke="{ink}" stroke-width="1.8" stroke-dasharray="2 3"/>']
    parts.append(f'<path d="M60 80 c-4 -6 -12 -6 -16 -1 c-2 3 -6 3 -8 0 c2 7 12 9 18 5 c3 -2 5 -3 6 -4 c1 1 3 2 6 4 '
                 f'c6 4 16 2 18 -5 c-2 3 -6 3 -8 0 c-4 -5 -12 -5 -16 1 z" fill="{ink}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -4 120 112">{"".join(parts)}</svg>'


def cloche(mood: str = "skeptical", size: int = 150, tone: str | None = None) -> str:
    """Cloche as an <img> (st.html strips inline SVG), in the active look's colours."""
    from . import style

    data = base64.b64encode(cloche_svg(mood, tone or style.current_look()).encode()).decode()
    return (f'<img class="od-cloche" src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" '
            f'alt="{html.escape(CLOCHE_LABELS.get(mood, "Cloche"))}">')


def _polar(cx: float, cy: float, r: float, p: float) -> tuple[float, float]:
    """The point at share `p` (0 to 1) along the dial's 240-degree arc, which opens at the bottom."""
    angle = math.radians(150 + 240 * min(max(p, 0.0), 1.0))
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def _arc(cx: float, cy: float, r: float, p0: float, p1: float) -> str:
    (x0, y0), (x1, y1) = _polar(cx, cy, r, p0), _polar(cx, cy, r, p1)
    large = 1 if (p1 - p0) * 240 > 180 else 0
    return f"M{x0:.2f} {y0:.2f} A{r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def dial(probability: float, review: float, ai: float, verdict: str) -> str:
    """Glass's AI likelihood dial, as an <img>: the three bands around the rim and a lit ring to the score.
    st.html strips inline SVG, so the dial goes in as an image with the active look's colours written in."""
    from . import style

    c = {**look_tones(), **{k: v for k, v in style.tokens().items() if k in ("accent-2", "panel-solid", "bg")}}
    mark = c[BANDS[verdict]]
    cx, cy, r = 115.0, 104.0, 78.0
    p = min(max(probability, 0.0), 1.0)
    gap = 0.012
    zones = (("human", 0, review), ("review", review, ai), ("ai", ai, 1))
    sans = "Geist, -apple-system, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif"
    mono = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"
    parts = [f'<path d="{_arc(cx, cy, r, 0, 1)}" stroke="{c["track"]}" stroke-width="14" fill="none" '
             'stroke-linecap="round"/>']
    for name, start, end in zones:
        parts.append(f'<path d="{_arc(cx, cy, r + 13, start + gap / 2, end - gap / 2)}" stroke="{c[name]}" '
                     'stroke-width="3" fill="none" stroke-linecap="round" opacity="0.8"/>')
    if p > 0.004:
        parts.append(f'<path d="{_arc(cx, cy, r, 0, p)}" stroke="url(#fill)" stroke-width="14" fill="none" '
                     'stroke-linecap="round" filter="url(#glow)"/>')
    kx, ky = _polar(cx, cy, r, p)
    parts.append(f'<circle cx="{kx:.2f}" cy="{ky:.2f}" r="9" fill="{c["panel-solid"]}" stroke="{mark}" '
                 'stroke-width="3.5"/>')
    number, number_y = f"{probability * 100:.0f}", cy + 14
    for cut in (review, ai):
        lx, ly = _polar(cx, cy, r + 27, cut)
        parts.append(f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle" font-family="{mono}" font-size="9.5" '
                     f'font-weight="600" fill="{c["muted"]}">{cut:.0%}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 230 196" width="230" height="196">'
           '<defs><linearGradient id="fill" x1="0" y1="1" x2="1" y2="0">'
           f'<stop offset="0" stop-color="{c["accent-2"]}"/><stop offset="1" stop-color="{mark}"/></linearGradient>'
           '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.5" '
           'result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
           f'{"".join(parts)}'
           f'<text x="{cx}" y="{number_y}" text-anchor="middle" font-family="{html.escape(sans)}" fill="{c["ink"]}">'
           f'<tspan font-size="46" font-weight="700" letter-spacing="-1.5">{number}</tspan>'
           f'<tspan font-size="18" font-weight="600" fill="{c["muted"]}" dx="2">%</tspan></text>'
           f'<text x="{cx}" y="{number_y + 20}" text-anchor="middle" font-family="{mono}" font-size="9" '
           f'font-weight="600" letter-spacing="1.6" fill="{c["muted"]}">AI LIKELIHOOD</text></svg>')
    data = base64.b64encode(svg.encode()).decode()
    label = f"AI likelihood {probability:.0%}; needs review above {review:.0%}, likely AI above {ai:.0%}"
    return f'<img class="od-dial" src="data:image/svg+xml;base64,{data}" alt="{label}">'


# The Adversarial Lab's priorities, clockwise from the top of the diamond.
CORNERS = {"detector": (0, -1, "DETECTOR"), "style": (1, 0, "STYLE"), "speed": (0, 1, "SPEED"),
           "quota": (-1, 0, "QUOTA")}


def priorities_chart(points: dict[str, int], most: int) -> str:
    """The points spent on each priority as a diamond, a corner each, as an <img> in the active look's
    colours (st.html strips inline SVG)."""
    c = look_tones()
    cx, cy, r = 95.0, 75.0, 50.0
    mono = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

    def corner(measure: str, share: float) -> str:
        dx, dy, _ = CORNERS[measure]
        return f"{cx + dx * r * share:.1f},{cy + dy * r * share:.1f}"

    parts = [f'<polygon points="{" ".join(corner(m, ring) for m in CORNERS)}" fill="none" stroke="{c["track"]}" '
             f'stroke-width="{1.4 if ring == 1 else 1}"/>' for ring in (1 / 3, 2 / 3, 1)]
    parts += [f'<line x1="{cx}" y1="{cy}" x2="{corner(m, 1).split(",")[0]}" y2="{corner(m, 1).split(",")[1]}" '
              f'stroke="{c["track"]}"/>' for m in CORNERS]
    shares = {m: max(points.get(m, 0) / most, 0.04) for m in CORNERS}  # a corner with no points still shows
    parts.append(f'<polygon points="{" ".join(corner(m, shares[m]) for m in CORNERS)}" fill="{c["accent"]}" '
                 f'fill-opacity="0.28" stroke="{c["accent"]}" stroke-width="2" stroke-linejoin="round"/>')
    parts += [f'<circle cx="{corner(m, shares[m]).split(",")[0]}" cy="{corner(m, shares[m]).split(",")[1]}" r="3" '
              f'fill="{c["accent"]}"/>' for m in CORNERS]
    for dx, dy, label in CORNERS.values():
        x, y = cx + dx * (r + 8), cy + dy * (r + 8) + (3 if dx else 0) + (-2 if dy < 0 else 9 if dy > 0 else 0)
        anchor = "middle" if not dx else "start" if dx > 0 else "end"
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{html.escape(mono)}" '
                     f'font-size="9.5" font-weight="600" letter-spacing="1" fill="{c["muted"]}">{label}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 190 150" width="190" height="150">'
           f'{"".join(parts)}</svg>')
    data = base64.b64encode(svg.encode()).decode()
    label = ", ".join(f"{CORNERS[m][2].lower()} {points.get(m, 0)} of {most}" for m in CORNERS)
    return f'<img class="od-priorities" src="data:image/svg+xml;base64,{data}" alt="Priorities: {label}">'


def order_ticket(title: str, timing: str, stations: list[tuple[str, str, str]], done: float, foot: str) -> None:
    """A running batch as an order ticket: the order, one line per station of the pipeline (state "done", "now"
    or "todo", title, note), how far it has got, and a line at the foot. Cloche watches from his corner and the
    kitchen timer runs under the ticket (ui/redteam.py `_ticket`)."""
    marks = {"done": "\u2713", "now": "\u25cf", "todo": "\u25cb"}
    rows = "".join(f'<li class="{state}"><i aria-hidden="true">{marks[state]}</i><div><b>{html.escape(name)}</b>'
                   f'<span>{html.escape(note)}</span></div></li>' for state, name, note in stations)
    st.html(f"""
<div class="od-ticket" role="status">
  <div class="od-ticket-head"><div><div class="od-eyebrow">Order ticket</div>
    <b>{html.escape(title)}</b><span>{html.escape(timing)}</span></div></div>
  <div class="od-rule"></div>
  <ol>{rows}</ol>
  <div class="od-ticket-bar" role="progressbar" aria-valuenow="{done * 100:.0f}" aria-valuemin="0"
    aria-valuemax="100"><i style="width: {min(max(done, 0), 1) * 100:.1f}%"></i></div>
  <small>{html.escape(foot)}</small>
</div>""")


def verdict_badge(verdict: str | None) -> None:
    label, color, icon = VERDICTS.get(verdict or "", ("Not scored", "gray", ":material/remove:"))
    st.badge(label, color=color, icon=icon)


def verdict_card(verdict: str, probability: float, thresholds: dict[str, float], calibrated: bool,
                 foot: str) -> None:
    """The document verdict, its AI likelihood, and where that falls against the two cut-offs."""
    label, color, _ = VERDICTS[verdict]
    review, ai = thresholds["review"], thresholds["ai"]
    if not calibrated:
        meaning = "Uncalibrated: the cut-offs are placeholders, with no measured false-positive rate behind them."
    else:
        meaning = {
            "likely_ai": "Above the cut-off that flags about 1 in 100 human texts.",
            "needs_review": "Between the two cut-offs: worth a closer read, not a conclusion.",
            "likely_human": "Below the cut-off that flags about 5 in 100 human texts.",
        }[verdict]
    from . import style

    if style.bistro():  # Cloche reacts from his corner (ui/kitchen.py); the band's needle shows where the score sits
        p = min(max(probability, 0.0), 1.0)
        st.html(f"""
<div class="od-verdict" style="--tone: var(--od-{color})">
  <div class="od-critic" data-mood="{MOODS[verdict]}">
    <div>
      <div class="od-eyebrow">The critic says</div>
      <div class="od-says"><div class="od-label">{label}</div><span>{probability:.0%} AI likelihood</span></div>
      <div class="od-meaning">{meaning}</div>
      <div class="od-band" role="img" aria-label="AI likelihood {probability:.0%} against the cut-offs at
        {review:.0%} and {ai:.0%}"><i class="human" style="flex: 0 0 {review * 100:.1f}%"></i>
        <i class="review" style="flex: 0 0 calc({(ai - review) * 100:.1f}% - 8px)"></i><i class="ai"></i>
        <b style="left: {p * 100:.1f}%"></b></div>
      <div class="od-band-ends"><span>Likely human &lt; {review:.0%}</span><span>Needs review</span>
        <span>Likely AI &gt; {ai:.0%}</span></div>
    </div>
  </div>
  <div class="od-foot">{html.escape(foot)}</div>
</div>""")
        return
    zones = (("likely_human", "human", f"Likely human < {review:.0%}"),
             ("needs_review", "review", f"Needs review {review:.0%}\u2013{ai:.0%}"),
             ("likely_ai", "ai", f"Likely AI > {ai:.0%}"))
    chips = "".join(f'<span class="{"on" if band == verdict else ""}"><i style="background: var(--od-{tone})"></i>'
                    f'{text}</span>' for band, tone, text in zones)
    gauge = dial(probability, review, ai, verdict)
    st.html(f"""
<div class="od-verdict" style="--tone: var(--od-{color}); --mark: var(--od-{BANDS[verdict]})">
  <div class="od-top">
    <div>
      <div class="od-eyebrow">Verdict</div>
      <div class="od-label">{label}</div>
      <div class="od-meaning">{meaning}</div>
      <div class="od-zones">{chips}</div>
    </div>
    {gauge}
  </div>
  <div class="od-foot">{html.escape(foot)}</div>
</div>""")


@dataclass(frozen=True)
class Reading:
    """One engine's reading for the readings panel. `share` (0 to 1) draws a meter; None shows the
    value alone, for scores that aren't probabilities."""

    name: str
    value: str
    note: str
    share: float | None = None


def engine_panel(readings: list[Reading], stats: list[tuple[str, str]],
                 total: tuple[str, str, str] | None = None, hygiene: str | None = None) -> None:
    """Each engine's own reading as a labelled meter, with a row of counts underneath. In Bistro it prints
    as the critic's bill instead: each reading and count on a dotted line, then `total` (the AI likelihood,
    the tone to print it in, and a note under it), with the `hygiene` line in a panel beside it, as tall."""
    from . import style

    if style.bistro():
        lines = [f'<div class="od-line"><span>{html.escape(r.name)}</span><i></i><b>{html.escape(r.value)}</b></div>'
                 for r in readings]
        lines += [f'<div class="od-line"><span>{html.escape(name.capitalize())}</span><i></i>'
                  f'<b>{html.escape(value)}</b></div>' for value, name in stats]
        end = ""
        if total:
            value, tone, note = total
            end = (f'<div class="od-rule"></div><div class="od-line total"><span>TOTAL · AI likelihood</span><i></i>'
                   f'<b style="color: var(--od-{tone})">{html.escape(value)}</b></div>'
                   f'<small>{html.escape(note)}</small>')
        bill = f'<div class="od-bill"><div class="od-eyebrow">The bill</div>{"".join(lines)}{end}</div>'
        if hygiene is not None:
            bill = (f'<div class="od-billrow">{bill}<div class="od-hygiene"><div class="od-eyebrow">Kitchen hygiene'
                    f'</div><div>{html.escape(hygiene)}</div></div></div>')
        st.html(bill)
        return
    rows = []
    for reading in readings:
        meter = (f'<div class="od-meter"><i style="width:{min(max(reading.share, 0), 1) * 100:.1f}%"></i></div>'
                 if reading.share is not None else "")
        rows.append(f'<div class="od-engine"><div class="od-engine-head"><span class="od-engine-name">'
                    f'{html.escape(reading.name)}</span><span class="od-engine-value">{html.escape(reading.value)}'
                    f'</span></div>{meter}<div class="od-note">{html.escape(reading.note)}</div></div>')
    counts = "".join(f"<div><b>{html.escape(value)}</b><span>{html.escape(name)}</span></div>" for value, name in stats)
    st.html(f'<div class="od-panel"><div class="od-eyebrow">Engine readings</div>{"".join(rows)}'
            f'<div class="od-stats">{counts}</div></div>')


def _depth(probability: float, band: str, thresholds: dict[str, float]) -> float:
    """How far a score sits into its band: 0 at the cut-off it crossed, 1 at the band's far end."""
    review, ai = thresholds["review"], thresholds["ai"]
    if band == "likely_ai":
        depth = (probability - ai) / (1 - ai) if ai < 1 else 1.0
    elif band == "needs_review":
        depth = (probability - review) / (ai - review) if ai > review else 1.0
    else:
        depth = (review - probability) / review if review > 0 else 1.0
    return min(max(depth, 0.0), 1.0)


# Highlight opacity at the near and far edge of each band, so deeper scores show stronger.
SHADES = {"likely_human": (12, 28), "needs_review": (20, 40), "likely_ai": (24, 46)}


def _runs(sentences: list, bands: list[str]) -> list[tuple[int, int, str]]:
    """Consecutive sentences in the same band, as (first, last, band) with 0-based indexes."""
    runs = []
    for i, band in enumerate(bands):
        if runs and runs[-1][2] == band:
            runs[-1] = (runs[-1][0], i, band)
        else:
            runs.append((i, i, band))
    return runs


def highlighted_text(text: str, sentences: list, bands: list[str], thresholds: dict[str, float],
                     flagged_only: bool = False, pickable: bool = False) -> str:
    """The text as HTML, every sentence tinted by its band and shaded by how deep into it its score
    sits, under a legend and a map of the whole text. Hovering shows a sentence's score; clicking the
    map jumps to that stretch. Likely human sentences stay plain when `flagged_only` is set. `pickable`
    leaves out the script, for pickable_highlights, whose own script also lets the reader pick sentences."""
    root = "od-" + hashlib.sha1(f"{text}{bands}{flagged_only}".encode()).hexdigest()[:10]
    words = [max(count_words(sentence.text), 1) for sentence in sentences]
    total = sum(words)

    parts, cursor = [], 0
    for number, (sentence, band) in enumerate(zip(sentences, bands, strict=True), 1):
        parts.append(html.escape(text[cursor : sentence.start]).replace("\n", "<br>"))
        tip = html.escape(f"{sentence.ai_probability:.0%} AI likelihood\n{VERDICTS[band][0]} · sentence {number}",
                          quote=True)
        body = html.escape(sentence.text)
        if band in BANDS and not (flagged_only and band == "likely_human"):
            low, high = SHADES[band]
            alpha = low + (high - low) * _depth(sentence.ai_probability, band, thresholds)
            parts.append(f'<span class="od-hl {BANDS[band]}" id="{root}-s{number}" data-n="{number}" '
                         f'style="--a:{alpha:.0f}%" data-tip="{tip}">{body}</span>')
        else:
            parts.append(f'<span class="od-hl" id="{root}-s{number}" data-n="{number}" data-tip="{tip}">{body}</span>')
        cursor = sentence.end
    parts.append(html.escape(text[cursor:]).replace("\n", "<br>"))

    keys = []
    for band in reversed(BANDS):
        share = sum(w for w, b in zip(words, bands, strict=True) if b == band) / total
        keys.append(f'<span class="od-key"><i class="{BANDS[band]}"></i>{VERDICTS[band][0]} <b>{share:.0%}</b></span>')
    hint = ("click sentences to pick them for revising, shift-click for a run" if pickable
            else "hover a sentence for its score, click the map to jump")
    legend = f'<div class="od-legend">{"".join(keys)}<span class="od-hint">share of the words · {hint}</span></div>'

    runs = []
    for first, last, band in _runs(sentences, bands):
        scores = [s.ai_probability for s in sentences[first : last + 1]]
        where = f"Sentence {first + 1}" if first == last else f"Sentences {first + 1}\u2013{last + 1}"
        spread = (f"{scores[0]:.0%}" if first == last else f"{min(scores):.0%}\u2013{max(scores):.0%}")
        tip = html.escape(f"{where} · {VERDICTS[band][0]}\n{spread} AI likelihood", quote=True)
        tone = "plain" if flagged_only and band == "likely_human" else BANDS.get(band, "plain")
        runs.append(f'<div class="od-run {tone}" style="flex:{sum(words[first : last + 1])}" '
                    f'data-tip="{tip}" data-jump="{root}-s{first + 1}"></div>')
    page_map = (f'<div class="od-map" role="img" aria-label="Map of the text by sentence band">{"".join(runs)}</div>'
                '<div class="od-ends"><span>Start</span><span>End</span></div>')

    kind = ' class="od-pickable"' if pickable else ""
    body = (f'<div id="{root}"{kind}>{legend}{page_map}<div class="od-text">{"".join(parts)}</div>'
            '<div class="od-tip" hidden></div></div>')
    return body if pickable else body + f'<script>{HIGHLIGHT_JS.replace("ROOT", root)}</script>'


# The highlights as a Streamlit component (st.components.v2, in the page itself, so the app's styles apply): the
# tooltip and map of HIGHLIGHT_JS, and a click on a sentence picks it or lets it go, shift-click a run from the last
# one clicked. The picked sentence numbers go back to Python as the "selected" state, and stay picked across reruns
# by the component's key. A click that ends a drag to select words for copying picks nothing.
PICK_JS = """
const picked = new Map();

export default function (component) {
  const { data, key, parentElement, setStateValue } = component;
  let host = parentElement.querySelector(":scope > .od-pick-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "od-pick-host";
    parentElement.appendChild(host);
  }
  if (host.dataset.digest !== data.digest) {
    host.innerHTML = data.html;
    host.dataset.digest = data.digest;
  }
  if (!picked.has(key)) picked.set(key, { chosen: new Set(data.selected || []), last: null });
  const memory = picked.get(key);
  const paint = () => host.querySelectorAll("[data-n]").forEach((el) => {
    el.classList.toggle("od-picked", memory.chosen.has(Number(el.dataset.n)));
    el.setAttribute("aria-pressed", memory.chosen.has(Number(el.dataset.n)) ? "true" : "false");
  });
  paint();
  const tip = host.querySelector(".od-tip");
  const move = (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) { tip.hidden = true; return; }
    if (tip.textContent !== target.dataset.tip) tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const below = event.clientY + 18 + tip.offsetHeight < window.innerHeight;
    tip.style.left = Math.max(8, Math.min(event.clientX + 12, window.innerWidth - tip.offsetWidth - 8)) + "px";
    tip.style.top = (below ? event.clientY + 18 : event.clientY - tip.offsetHeight - 12) + "px";
  };
  const leave = () => { tip.hidden = true; };
  // Shift-click would extend the browser's text selection; here it picks a run of sentences instead.
  const down = (event) => { if (event.shiftKey && event.target.closest("[data-n]")) event.preventDefault(); };
  const click = (event) => {
    const run = event.target.closest("[data-jump]");
    if (run) {
      const sentence = document.getElementById(run.dataset.jump);
      if (!sentence) return;
      sentence.scrollIntoView({ block: "center", behavior: "smooth" });
      sentence.classList.remove("od-flash");
      void sentence.offsetWidth;
      sentence.classList.add("od-flash");
      return;
    }
    const sentence = event.target.closest("[data-n]");
    if (!sentence) return;
    if (event.shiftKey) window.getSelection().removeAllRanges();
    else if (String(window.getSelection()).trim()) return;  // the end of a drag to copy words
    const number = Number(sentence.dataset.n);
    if (event.shiftKey && memory.last !== null) {
      const on = !memory.chosen.has(number);
      const [low, high] = [Math.min(memory.last, number), Math.max(memory.last, number)];
      for (let n = low; n <= high; n++) on ? memory.chosen.add(n) : memory.chosen.delete(n);
    } else if (memory.chosen.has(number)) {
      memory.chosen.delete(number);
    } else {
      memory.chosen.add(number);
    }
    memory.last = number;
    paint();
    setStateValue("selected", [...memory.chosen].sort((a, b) => a - b));
  };
  host.addEventListener("mousemove", move);
  host.addEventListener("mouseleave", leave);
  host.addEventListener("click", click);
  host.addEventListener("mousedown", down);
  window.addEventListener("scroll", leave, { capture: true, passive: true });
  return () => {
    host.removeEventListener("mousemove", move);
    host.removeEventListener("mouseleave", leave);
    host.removeEventListener("click", click);
    host.removeEventListener("mousedown", down);
    window.removeEventListener("scroll", leave, { capture: true });
  };
}
"""
_picker = None


def pickable_highlights(page: str, key: str) -> list[int]:
    """`page` (highlighted_text with `pickable`) drawn so the reader can pick sentences by clicking them. Gives the
    picked sentences' indexes, from 0, in order. A new `key` starts with none picked."""
    global _picker
    if _picker is None:
        from streamlit.components.v2 import component

        _picker = component("od_pickable_highlights", js=PICK_JS, isolate_styles=False)
    digest = hashlib.sha1(page.encode()).hexdigest()
    result = _picker(data={"html": page, "digest": digest}, key=key, default={"selected": []},
                     on_selected_change=lambda: None)
    return [number - 1 for number in (result.get("selected") or []) if isinstance(number, int) and number > 0]


def _clip(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dress(chart: alt.LayerChart) -> alt.LayerChart:
    """Lets the panel behind a chart show through, in the active look's type."""
    from . import style

    font = ("'JetBrains Mono', 'SF Mono', Menlo, monospace" if style.bistro()
            else "Geist, Inter, -apple-system, system-ui, sans-serif")
    tones = look_tones()
    return chart.configure(background="transparent", font=font).configure_view(stroke=None).configure_axis(
        labelColor=tones["muted"], titleColor=tones["muted"], domainColor=tones["track"], tickColor=tones["track"])


def _band_color() -> alt.Scale:
    tones = look_tones()
    return alt.Scale(domain=[VERDICTS[band][0] for band in BANDS], range=[tones[tone] for tone in BANDS.values()])


def sentence_chart(sentences: list, bands: list[str], thresholds: dict[str, float]) -> alt.LayerChart:
    """Every sentence as a bar as wide as its word count and as tall as its AI likelihood, in reading
    order, against the two cut-offs."""
    tones = look_tones()
    rows, position = [], 0
    for number, (sentence, band) in enumerate(zip(sentences, bands, strict=True), 1):
        words = max(count_words(sentence.text), 1)
        rows.append({"sentence": number, "start": position, "end": position + words, "p": sentence.ai_probability,
                     "band": VERDICTS[band][0], "text": _clip(sentence.text)})
        position += words
    # The y axis is ticked at the cut-offs, which labels the dashed lines without covering any bars.
    ticks = [0, thresholds["review"], thresholds["ai"], 1]
    x = alt.X("start:Q", title="Position in the text (words)", scale=alt.Scale(domain=[0, position], nice=False),
              axis=alt.Axis(grid=False))
    y = alt.Y("p:Q", title="AI likelihood", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", values=ticks, grid=False))
    color = alt.Color("band:N", scale=_band_color(),
                      legend=alt.Legend(orient="top", title=None, symbolType="square", symbolOpacity=1))
    tooltip = [alt.Tooltip("sentence:Q", title="Sentence"), alt.Tooltip("p:Q", title="AI likelihood", format=".0%"),
               alt.Tooltip("band:N", title="Band"), alt.Tooltip("text:N", title="Text")]
    # A light wash for each sentence with a solid line along its top, like a stepped area chart.
    base = alt.Chart(pd.DataFrame(rows))
    wash = base.mark_bar(fillOpacity=0.28, stroke=tones["surface"], strokeWidth=1).encode(
        x=x, x2="end:Q", y=y, y2=alt.datum(0), color=color, tooltip=tooltip)
    tops = base.mark_rule(strokeWidth=2.5).encode(x=x, x2="end:Q", y=y, color=color, tooltip=tooltip)
    cutoffs = pd.DataFrame([{"y": thresholds["review"]}, {"y": thresholds["ai"]}])
    rules = alt.Chart(cutoffs).mark_rule(strokeDash=[4, 4], color=tones["muted"]).encode(y="y:Q")
    return _dress(alt.layer(wash, rules, tops).properties(height=320))


def score_bars(rows: list[dict], thresholds: dict[str, float]) -> alt.LayerChart | None:
    """Sections or blocks in document order as horizontal bars of AI likelihood, coloured by verdict,
    with the cut-offs as dashed lines. Each row has `label`, `probability` (None when the detector
    abstained) and `verdict`. None when nothing was scored."""
    if not any(row["probability"] is not None for row in rows):
        return None
    tones = look_tones()
    seen: dict[str, int] = {}
    data = []
    for row in rows:
        seen[row["label"]] = seen.get(row["label"], 0) + 1
        key = row["label"] if seen[row["label"]] == 1 else f"{row['label']} ({seen[row['label']]})"
        scored = row["probability"] is not None and row["verdict"] in BANDS
        data.append({"key": _clip(key, 60), "p": row["probability"] if scored else None,
                     "tip": row["probability"] if scored else 0.0,
                     "band": VERDICTS[row["verdict"]][0] if scored else "Not scored",
                     "value": f"{row['probability']:.0%}" if scored else
                     ("not prose" if row["verdict"] == "not_prose" else "too short to score")})
    frame = pd.DataFrame(data)
    y = alt.Y("key:N", title=None, scale=alt.Scale(domain=list(frame["key"])), axis=alt.Axis(labelLimit=280))
    x = alt.X("p:Q", title="AI likelihood", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False))
    tooltip = [alt.Tooltip("key:N", title="Section"), alt.Tooltip("value:N", title="AI likelihood"),
               alt.Tooltip("band:N", title="Verdict")]
    base = alt.Chart(frame)
    bars = base.transform_filter("isValid(datum.p)").mark_bar(cornerRadiusEnd=4, size=18).encode(
        x=x, y=y, color=alt.Color("band:N", scale=_band_color(), legend=alt.Legend(orient="top", title=None)),
        tooltip=tooltip)
    values = base.mark_text(align="left", dx=5, fontSize=12, color=tones["ink"]).encode(
        x=alt.X("tip:Q", scale=alt.Scale(domain=[0, 1])), y=y, text="value:N", tooltip=tooltip)
    rules = alt.Chart(pd.DataFrame([{"x": thresholds["review"]}, {"x": thresholds["ai"]}])).mark_rule(
        strokeDash=[4, 4], color=tones["muted"]).encode(x="x:Q")
    return _dress(alt.layer(bars, rules, values).properties(height=30 * len(frame) + 90))


def contribution_chart(parts: dict[str, float]) -> alt.LayerChart:
    """How far each engine pushed the combined score toward AI (right) or human (left), in log-odds."""
    tones = look_tones()
    frame = pd.DataFrame([{"engine": name, "value": value, "label": f"{value:+.2f}",
                           "direction": "Toward AI" if value > 0 else "Toward human"} for name, value in parts.items()])
    reach = max(abs(frame["value"]).max() * 1.3, 0.5)
    x = alt.X("value:Q", title="Push on the score (log-odds)", scale=alt.Scale(domain=[-reach, reach]),
              axis=alt.Axis(grid=False))
    y = alt.Y("engine:N", title=None, sort=None)
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, size=20).encode(
        x=x, y=y,
        color=alt.Color("direction:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Toward human", "Toward AI"], range=[tones["human"], tones["ai"]])),
        tooltip=[alt.Tooltip("engine:N", title="Engine"), alt.Tooltip("label:N", title="Push")])
    zero = alt.Chart(pd.DataFrame([{"x": 0}])).mark_rule(color=tones["muted"]).encode(x="x:Q")
    right = alt.Chart(frame).transform_filter("datum.value >= 0").mark_text(
        align="left", dx=5, color=tones["ink"]).encode(x=x, y=y, text="label:N")
    left = alt.Chart(frame).transform_filter("datum.value < 0").mark_text(
        align="right", dx=-5, color=tones["ink"]).encode(x=x, y=y, text="label:N")
    return _dress(alt.layer(bars, zero, right, left).properties(height=44 * len(frame) + 100))


def keep(key: str, default) -> str:
    """Set a widget's starting value and return its key. Keys that start with "keep:" survive a switch
    to the other page (app.py writes them back on every run). Such a widget must take its starting
    value from here, not from a `value` argument, which Streamlit would warn about."""
    st.session_state.setdefault(key, default)
    return key


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes

    @property
    def suffix(self) -> str:
        return "." + self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""


def persistent_upload(label: str, key: str, types: tuple[str, ...], help: str | None = None) -> Upload | None:
    """A file uploader whose file stays chosen when the user switches pages and back. Streamlit
    empties an uploader the page didn't draw, so the file is remembered, and shown in its place,
    until the user removes it."""
    stored_key = f"upload:{key}"

    def remember() -> None:
        file = st.session_state[key]
        st.session_state[stored_key] = Upload(file.name, file.getvalue()) if file else None

    stored = st.session_state.get(stored_key)
    if stored and st.session_state.get(key) is None:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/description: **{stored.name}** · {len(stored.data) / 1024:,.0f} KB")
            if st.button("Remove", key=f"{key}:remove", icon=":material/close:", type="tertiary"):
                st.session_state[stored_key] = None
                st.rerun()
        return stored
    st.file_uploader(label, type=types, key=key, on_change=remember, help=help)
    return st.session_state.get(stored_key)
