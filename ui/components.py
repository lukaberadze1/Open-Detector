"""Pieces both pages use: the theme's extra styles, verdict display, sentence highlights, and widgets
that keep their state when the user switches pages."""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

# Label, Streamlit color name and icon for each verdict.
VERDICTS = {
    "likely_ai": ("Likely AI", "red", ":material/smart_toy:"),
    "needs_review": ("Needs review", "orange", ":material/flag:"),
    "likely_human": ("Likely human", "green", ":material/person:"),
    "insufficient_text": ("Too short", "gray", ":material/short_text:"),
    "not_prose": ("Not prose", "gray", ":material/format_list_bulleted:"),
}
# Text-safe tones for each color in the light and dark palettes (.streamlit/config.toml).
TONES = {
    "light": {"red": "#DC2626", "orange": "#B45309", "green": "#15803D", "gray": "#64748B", "surface": "#FFFFFF"},
    "dark": {"red": "#F87171", "orange": "#FBBF24", "green": "#4ADE80", "gray": "#94A3B8", "surface": "#0D0F16"},
}

CSS = """
.od-verdict { border: 1px solid color-mix(in srgb, var(--tone) 35%, transparent);
  background: color-mix(in srgb, var(--tone) 7%, transparent); border-radius: 0.9rem; padding: 1.2rem 1.4rem 1rem; }
.od-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.od-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; opacity: 0.65; }
.od-label { font-size: 1.9rem; font-weight: 700; line-height: 1.2; color: var(--tone); margin: 0.1rem 0 0.25rem; }
.od-meaning { font-size: 0.92rem; opacity: 0.85; max-width: 34rem; }
.od-score { text-align: right; white-space: nowrap; }
.od-number { font-size: 2.6rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.od-number small { font-size: 1.3rem; font-weight: 600; opacity: 0.7; margin-left: 1px; }
.od-gauge { position: relative; display: flex; height: 10px; margin: 1.2rem 0 0.45rem; }
.od-zone { height: 100%; }
.od-zone:first-child { border-radius: 999px 0 0 999px; }
.od-zone:last-of-type { border-radius: 0 999px 999px 0; }
.od-zone.human { background: color-mix(in srgb, var(--od-green) 30%, transparent); }
.od-zone.review { background: color-mix(in srgb, var(--od-orange) 30%, transparent); }
.od-zone.ai { background: color-mix(in srgb, var(--od-red) 30%, transparent); }
.od-marker { position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%; background: var(--tone);
  transform: translate(-50%, -50%); box-shadow: 0 0 0 3px var(--od-surface), 0 1px 6px rgba(0, 0, 0, 0.25); }
.od-zones { display: flex; font-size: 0.74rem; opacity: 0.7; }
.od-zones span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 0.4rem; }
.od-foot { font-size: 0.8rem; opacity: 0.65; margin-top: 0.8rem; }
.od-text { line-height: 1.85; font-size: 1rem; max-height: 34rem; overflow-y: auto; padding: 1rem 1.25rem;
  border: 1px solid color-mix(in srgb, currentColor 14%, transparent); border-radius: 0.75rem; }
.od-sample { white-space: pre-wrap; max-height: 22rem; font-size: 0.95rem; line-height: 1.7; }
.od-hl { border-radius: 4px; padding: 1px 2px; -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.od-hl.likely_ai { background: color-mix(in srgb, var(--od-red) 28%, transparent); }
.od-hl.needs_review { background: color-mix(in srgb, var(--od-orange) 28%, transparent); }
.od-legend { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.82rem; opacity: 0.85; margin-bottom: 0.6rem; }
.od-legend i { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 3px; margin-right: 0.35rem;
  vertical-align: -1px; }
"""


def theme_type() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


def apply_theme() -> None:
    """Styles for the app's own HTML (verdict card, highlights), in the active palette's tones."""
    tones = TONES[theme_type()]
    variables = "".join(f"--od-{name}: {value};" for name, value in tones.items())
    st.html(f"<style>:root {{{variables}}}{CSS}</style>")


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
    zones = (("human", "Likely human", review), ("review", "Needs review", ai - review), ("ai", "Likely AI", 1 - ai))
    bars = "".join(f'<div class="od-zone {name}" style="flex:{width:.4f}"></div>' for name, _, width in zones)
    names = "".join(f'<span style="flex:{width:.4f}">{text}</span>' for _, text, width in zones)
    st.html(f"""
<div class="od-verdict" style="--tone: var(--od-{color})">
  <div class="od-top">
    <div>
      <div class="od-eyebrow">Verdict</div>
      <div class="od-label">{label}</div>
      <div class="od-meaning">{meaning}</div>
    </div>
    <div class="od-score">
      <div class="od-number">{probability * 100:.0f}<small>%</small></div>
      <div class="od-eyebrow">AI likelihood</div>
    </div>
  </div>
  <div class="od-gauge" role="img" aria-label="AI likelihood {probability:.0%}; needs review above {review:.0%},
    likely AI above {ai:.0%}">{bars}<div class="od-marker" style="left:{min(max(probability, 0), 1) * 100:.1f}%"></div>
  </div>
  <div class="od-zones">{names}</div>
  <div class="od-foot">{html.escape(foot)}</div>
</div>""")


def highlighted_text(text: str, sentences: list, bands: list[str]) -> str:
    """The text as HTML, with each sentence in a flagged band highlighted and its score on hover."""
    parts, cursor = [], 0
    for sentence, band in zip(sentences, bands, strict=True):
        parts.append(html.escape(text[cursor : sentence.start]).replace("\n", "<br>"))
        body = html.escape(sentence.text)
        if band in ("likely_ai", "needs_review"):
            tip = html.escape(f"{sentence.ai_probability:.0%} AI likelihood", quote=True)
            body = f'<span class="od-hl {band}" title="{tip}">{body}</span>'
        parts.append(body)
        cursor = sentence.end
    parts.append(html.escape(text[cursor:]).replace("\n", "<br>"))
    legend = ('<div class="od-legend"><span><i style="background:color-mix(in srgb, var(--od-red) 45%, transparent)">'
              '</i>Likely AI</span><span><i style="background:color-mix(in srgb, var(--od-orange) 45%, transparent)">'
              "</i>Needs review</span><span>Hover a sentence for its score</span></div>")
    return f'{legend}<div class="od-text">{"".join(parts)}</div>'


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
