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


def look_tones() -> dict[str, str]:
    """The verdict tones for the active mode, with the active look's surfaces (ui/style.py) on top."""
    from . import style

    merged = dict(TONES[theme_type()])
    look = style.tokens()
    merged.update({"surface": look["panel-solid"], "ink": look["ink"], "muted": look["muted"], "track": look["track"],
                   "accent": look["accent"]})
    return merged


def apply_theme() -> None:
    """Styles for the whole app: the active look (ui/style.py), then the app's own HTML (verdict card,
    highlights) in the verdict tones."""
    from . import style

    variables = "".join(f"--od-{name}: {value};" for name, value in TONES[theme_type()].items()
                        if name not in ("ink", "muted", "track", "accent", "surface"))
    st.html(f"<style>{style.css()}:root {{{variables}--od-surface: var(--od-panel-solid);}}{CSS}</style>")


def _polar(cx: float, cy: float, r: float, p: float) -> tuple[float, float]:
    """The point at share `p` (0 to 1) along the dial's 240-degree arc, which opens at the bottom."""
    angle = math.radians(150 + 240 * min(max(p, 0.0), 1.0))
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def _arc(cx: float, cy: float, r: float, p0: float, p1: float) -> str:
    (x0, y0), (x1, y1) = _polar(cx, cy, r, p0), _polar(cx, cy, r, p1)
    large = 1 if (p1 - p0) * 240 > 180 else 0
    return f"M{x0:.2f} {y0:.2f} A{r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}"


def dial(probability: float, review: float, ai: float, verdict: str, instrument: bool) -> str:
    """The AI likelihood as a dial, as an <img>: the three bands around the rim and the score on it.
    Glass draws a thick lit ring; Instrument a tick-marked scale with a needle. st.html strips inline
    SVG, so the dial goes in as an image with the active look's colours written in."""
    from . import style

    c = {**look_tones(), **{k: v for k, v in style.tokens().items() if k in ("accent-2", "panel-solid", "bg")}}
    mark = c[BANDS[verdict]]
    cx, cy, r = 115.0, 104.0, 78.0
    p = min(max(probability, 0.0), 1.0)
    gap = 0.012
    zones = (("human", 0, review), ("review", review, ai), ("ai", ai, 1))
    sans = "Geist, -apple-system, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif"
    mono = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"
    parts = []
    if instrument:
        for name, start, end in zones:
            parts.append(f'<path d="{_arc(cx, cy, r, start + gap / 2, end - gap / 2)}" stroke="{c[name]}" '
                         'stroke-width="3" fill="none"/>')
        for i in range(41):
            share, major = i / 40, i % 10 == 0
            (x0, y0), (x1, y1) = _polar(cx, cy, r - 7, share), _polar(cx, cy, r - (17 if major else 11), share)
            parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{c["muted"]}" '
                         f'stroke-width="{1.4 if major else 0.8}" opacity="{0.9 if major else 0.55}"/>')
        for cut in (review, ai):
            (x0, y0), (x1, y1) = _polar(cx, cy, r + 4, cut), _polar(cx, cy, r - 20, cut)
            parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="{c["ink"]}" '
                         'stroke-width="1.2" stroke-dasharray="2 2"/>')
        tip, tail = _polar(cx, cy, r - 2, p), _polar(cx, cy, 14, (p + 0.5) % 1)
        parts.append(f'<line x1="{tail[0]:.2f}" y1="{tail[1]:.2f}" x2="{tip[0]:.2f}" y2="{tip[1]:.2f}" stroke="{mark}" '
                     f'stroke-width="2.2" stroke-linecap="round" filter="url(#glow)"/>'
                     f'<circle cx="{cx}" cy="{cy}" r="5" fill="{c["bg"]}" stroke="{mark}" stroke-width="2"/>')
        number, number_y, font = f"{probability * 100:.1f}", cy + 68, mono
    else:
        parts.append(f'<path d="{_arc(cx, cy, r, 0, 1)}" stroke="{c["track"]}" stroke-width="14" fill="none" '
                     'stroke-linecap="round"/>')
        for name, start, end in zones:
            parts.append(f'<path d="{_arc(cx, cy, r + 13, start + gap / 2, end - gap / 2)}" stroke="{c[name]}" '
                         'stroke-width="3" fill="none" stroke-linecap="round" opacity="0.8"/>')
        if p > 0.004:
            parts.append(f'<path d="{_arc(cx, cy, r, 0, p)}" stroke="url(#fill)" stroke-width="14" fill="none" '
                         'stroke-linecap="round" filter="url(#glow)"/>')
        kx, ky = _polar(cx, cy, r, p)
        parts.append(f'<circle cx="{kx:.2f}" cy="{ky:.2f}" r="9" fill="{c["panel-solid"]}" stroke="{mark}" '
                     'stroke-width="3.5"/>')
        number, number_y, font = f"{probability * 100:.0f}", cy + 14, sans
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
           f'<text x="{cx}" y="{number_y}" text-anchor="middle" font-family="{html.escape(font)}" fill="{c["ink"]}">'
           f'<tspan font-size="{30 if instrument else 46}" font-weight="700" letter-spacing="-1.5">{number}</tspan>'
           f'<tspan font-size="{14 if instrument else 18}" font-weight="600" fill="{c["muted"]}" dx="2">%</tspan>'
           '</text>'
           f'<text x="{cx}" y="{number_y + (16 if instrument else 20)}" text-anchor="middle" font-family="{mono}" '
           'font-size="9" '
           f'font-weight="600" letter-spacing="1.6" fill="{c["muted"]}">AI LIKELIHOOD</text></svg>')
    data = base64.b64encode(svg.encode()).decode()
    label = f"AI likelihood {probability:.0%}; needs review above {review:.0%}, likely AI above {ai:.0%}"
    return f'<img class="od-dial" src="data:image/svg+xml;base64,{data}" alt="{label}">'


# The Red team page's priorities, clockwise from the top of their diamond.
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

    zones = (("likely_human", "human", f"Likely human < {review:.0%}"),
             ("needs_review", "review", f"Needs review {review:.0%}\u2013{ai:.0%}"),
             ("likely_ai", "ai", f"Likely AI > {ai:.0%}"))
    chips = "".join(f'<span class="{"on" if band == verdict else ""}"><i style="background: var(--od-{tone})"></i>'
                    f'{text}</span>' for band, tone, text in zones)
    gauge = dial(probability, review, ai, verdict, instrument=style.current_look() == "instrument")
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


def engine_panel(readings: list[Reading], stats: list[tuple[str, str]]) -> None:
    """Each engine's own reading as a labelled meter, with a row of counts underneath."""
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
                     flagged_only: bool = False) -> str:
    """The text as HTML, every sentence tinted by its band and shaded by how deep into it its score
    sits, under a legend and a map of the whole text. Hovering shows a sentence's score; clicking the
    map jumps to that stretch. Likely human sentences stay plain when `flagged_only` is set."""
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
            parts.append(f'<span class="od-hl {BANDS[band]}" id="{root}-s{number}" style="--a:{alpha:.0f}%" '
                         f'data-tip="{tip}">{body}</span>')
        else:
            parts.append(f'<span class="od-hl" id="{root}-s{number}" data-tip="{tip}">{body}</span>')
        cursor = sentence.end
    parts.append(html.escape(text[cursor:]).replace("\n", "<br>"))

    keys = []
    for band in reversed(BANDS):
        share = sum(w for w, b in zip(words, bands, strict=True) if b == band) / total
        keys.append(f'<span class="od-key"><i class="{BANDS[band]}"></i>{VERDICTS[band][0]} <b>{share:.0%}</b></span>')
    legend = (f'<div class="od-legend">{"".join(keys)}<span class="od-hint">share of the words · hover a sentence '
              "for its score, click the map to jump</span></div>")

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

    return (f'<div id="{root}">{legend}{page_map}<div class="od-text">{"".join(parts)}</div>'
            f'<div class="od-tip" hidden></div></div><script>{HIGHLIGHT_JS.replace("ROOT", root)}</script>')


def _clip(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dress(chart: alt.LayerChart) -> alt.LayerChart:
    """Lets the panel behind a chart show through, in the active look's type."""
    from . import style

    font = ("'JetBrains Mono', 'SF Mono', Menlo, monospace" if style.current_look() == "instrument"
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
                     "value": f"{row['probability']:.0%}" if scored else "too short to score"})
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
