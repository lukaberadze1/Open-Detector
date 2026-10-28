"""Pieces both pages use: the theme's extra styles, verdict display, sentence highlights, charts, and
widgets that keep their state when the user switches pages."""

from __future__ import annotations

import hashlib
import html
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
.od-verdict { border: 1px solid color-mix(in srgb, var(--tone) 35%, transparent);
  background: color-mix(in srgb, var(--tone) 7%, transparent); border-radius: 0.9rem; padding: 1.2rem 1.4rem 1rem; }
.od-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.od-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; opacity: 0.65; }
.od-label { font-size: 1.9rem; font-weight: 700; line-height: 1.2; color: var(--tone); margin: 0.1rem 0 0.25rem; }
.od-meaning { font-size: 0.92rem; opacity: 0.85; max-width: 34rem; }
.od-score { text-align: right; white-space: nowrap; }
.od-number { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.od-number small { font-size: 1.3rem; font-weight: 600; opacity: 0.7; margin-left: 1px; }
.od-gauge { position: relative; display: flex; gap: 2px; height: 10px; margin: 1.2rem 0 0.45rem; }
.od-zone { height: 100%; }
.od-zone.human { background: color-mix(in srgb, var(--od-human) 40%, transparent); border-radius: 999px 0 0 999px; }
.od-zone.review { background: color-mix(in srgb, var(--od-review) 45%, transparent); }
.od-zone.ai { background: color-mix(in srgb, var(--od-ai) 45%, transparent); border-radius: 0 999px 999px 0; }
.od-marker { position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%; background: var(--mark);
  transform: translate(-50%, -50%); box-shadow: 0 0 0 3px var(--od-surface), 0 1px 6px rgba(0, 0, 0, 0.25); }
.od-zones { display: flex; gap: 2px; font-size: 0.74rem; opacity: 0.7; }
.od-zones span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 0.4rem; }
.od-foot { font-size: 0.8rem; opacity: 0.65; margin-top: 0.8rem; }

.od-panel { border: 1px solid color-mix(in srgb, currentColor 14%, transparent); border-radius: 0.9rem;
  padding: 1rem 1.2rem; }
.od-engine { margin-top: 0.8rem; }
.od-engine-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; }
.od-engine-name { font-weight: 600; font-size: 0.95rem; }
.od-engine-value { font-weight: 600; font-variant-numeric: tabular-nums; }
.od-meter { height: 6px; border-radius: 999px; background: var(--od-track); margin: 0.35rem 0 0.3rem;
  overflow: hidden; }
.od-meter i { display: block; height: 100%; border-radius: 999px; background: var(--od-accent); }
.od-note { font-size: 0.78rem; opacity: 0.65; line-height: 1.35; }
.od-stats { display: flex; gap: 1.6rem; margin-top: 0.9rem; padding-top: 0.75rem;
  border-top: 1px solid color-mix(in srgb, currentColor 12%, transparent); }
.od-stats b { display: block; font-size: 1.15rem; font-weight: 600; line-height: 1.3; }
.od-stats span { font-size: 0.75rem; opacity: 0.65; }

.od-text { line-height: 1.85; font-size: 1rem; max-height: 34rem; overflow-y: auto; padding: 1rem 1.25rem;
  border: 1px solid color-mix(in srgb, currentColor 14%, transparent); border-radius: 0.75rem; }
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
.od-map { display: flex; gap: 2px; height: 12px; }
.od-run { min-width: 3px; border-radius: 3px; cursor: pointer; transition: transform 0.12s; }
.od-run.human { background: var(--od-human); }
.od-run.review { background: var(--od-review); }
.od-run.ai { background: var(--od-ai); }
.od-run.plain { background: var(--od-track); }
.od-run:hover { transform: scaleY(1.6); }
.od-ends { display: flex; justify-content: space-between; font-size: 0.72rem; opacity: 0.6; margin: 0.3rem 0 0.9rem; }
.od-tip { position: fixed; z-index: 1000; pointer-events: none; max-width: 20rem; padding: 0.4rem 0.6rem;
  border-radius: 0.5rem; background: var(--od-ink); color: var(--od-surface); font-size: 0.8rem; line-height: 1.4;
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
<div class="od-verdict" style="--tone: var(--od-{color}); --mark: var(--od-{BANDS[verdict]})">
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


def _band_color() -> alt.Scale:
    tones = TONES[theme_type()]
    return alt.Scale(domain=[VERDICTS[band][0] for band in BANDS], range=[tones[tone] for tone in BANDS.values()])


def sentence_chart(sentences: list, bands: list[str], thresholds: dict[str, float]) -> alt.LayerChart:
    """Every sentence as a bar as wide as its word count and as tall as its AI likelihood, in reading
    order, against the two cut-offs."""
    tones = TONES[theme_type()]
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
    return alt.layer(wash, rules, tops).properties(height=320)


def score_bars(rows: list[dict], thresholds: dict[str, float]) -> alt.LayerChart | None:
    """Sections or blocks in document order as horizontal bars of AI likelihood, coloured by verdict,
    with the cut-offs as dashed lines. Each row has `label`, `probability` (None when the detector
    abstained) and `verdict`. None when nothing was scored."""
    if not any(row["probability"] is not None for row in rows):
        return None
    tones = TONES[theme_type()]
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
    return alt.layer(bars, rules, values).properties(height=30 * len(frame) + 90)


def contribution_chart(parts: dict[str, float]) -> alt.LayerChart:
    """How far each engine pushed the combined score toward AI (right) or human (left), in log-odds."""
    tones = TONES[theme_type()]
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
    return alt.layer(bars, zero, right, left).properties(height=44 * len(frame) + 100)


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
