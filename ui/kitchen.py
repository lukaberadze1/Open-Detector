"""Bistro's moving parts: Cloche in the corner of every page, the kitchen bell, the kitchen timer and the
ingredient rows of the Lab's priorities.

Streamlit draws a page afresh on every click, so an animation drawn with st.html starts over each time. These
are st.components.v2 components instead, drawn in the page itself (not an iframe), and they keep their own
elements across reruns:

- Cloche lives on the document's <body>, outside everything Streamlit draws, so a rerun or a page switch never
  touches him. Python only sends him cues: `cue()` stores the latest one in the session with a number, and
  `show()` hands it to the page, which plays a cue once, by its number, however many times it is sent. A cue
  sent while a long job runs (`show(...)` before the job) reaches the page at once, since Streamlit sends each
  element as the script draws it.
- The timer counts down in the browser from the seconds left it was last given, so it ticks smoothly between
  the batch page's two-second refreshes and while the detector holds the script.
- The bell is a recorded kitchen bell (ui/assets/bell.mp3), carried inside the component's script and played with
  the Web Audio API, with a synthesised one until it is decoded. Browsers play sound only after the reader has
  clicked on the page, and the bell always follows a button press.

All three are Bistro's; in Glass, Cloche stays hidden. Motion stops for readers who ask for reduced motion.
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import streamlit as st

from . import style

STATE = "od:stage"  # the latest cue: number, mood, line, bell
GREETED = "od:stage:greeted"
MOODS = ("happy", "skeptical", "shocked", "tasting")
# What Cloche says when he's clicked, in turn, on each page.
LINES = {
    "detect": ["I have tasted 1.4 million texts. Most were bland.", "Careful, the dome is hot.",
               "Moustache freshly waxed, thank you.", "Compliments to the chef. Never to the AI.",
               "I can taste a semicolon from across the room.", "Psst. Try the Lab. The kitchen is open."],
    "lab": ["A pinch of basil never hurt anyone.", "Six tomatoes? Bold. I like it.",
            "Moustache freshly waxed, thank you.", "The cooks are nervous when I watch.",
            "Careful, the dome is hot.", "Everything from this kitchen is labelled AI. House rules."],
    "settings": ["Change nothing sharp.", "Two keys? Twice the cooks.", "Slim build. Like my waistline.",
                 "The light switch is in the ⋮ menu. I checked.", "Textured. As a dome should be."],
}
# The verdict, as Cloche's face and what he says when it lands.
VERDICT_CUES = {
    "likely_human": ("happy", "Délicieux. A person cooked this."),
    "needs_review": ("skeptical", "Hm. Something is off. Monocle up."),
    "likely_ai": ("shocked", "Sacré bleu! This came out of a machine."),
    "insufficient_text": ("skeptical", "A nibble, not a meal. I need 150 words to judge."),
    "not_prose": ("skeptical", "Poetry? Lists? I only judge prose."),
}
# What else he has to say about a verdict: one is picked when it lands, and clicking him afterwards repeats it, then
# goes through the others, his face staying the verdict's.
VERDICT_LINES = {
    "likely_ai": ["Sacré bleu! This came out of a machine.", "I taste… silicon. With a hint of autocomplete.",
                  "Mon dieu. This was microwaved.", "Too smooth. Nobody writes this clean before coffee.",
                  "Delve? DELVE? Send it back to the kitchen.", "A robot cooked this. With a thesaurus."],
    "needs_review": ["Hm. Something is off. Monocle up.", "Half chef, half toaster, I'd say.",
                     "I'm torn. Like a croissant.", "Some bites are home-made. Some came out of a packet.",
                     "Let me look closer. The monocle never lies."],
    "likely_human": ["Délicieux. A person cooked this.", "Lumpy in all the right places. Human.",
                     "Home-made. Chef's kiss.", "Made by hand, with love and a little panic.",
                     "I can taste the second thoughts. Wonderful."],
    "insufficient_text": ["A nibble, not a meal. I need 150 words to judge.",
                          "That's an amuse-bouche. Bring me a main course.", "Is that all? I'm still hungry."],
    "not_prose": ["Poetry? Lists? I only judge prose.", "Lovely shopping list. I review dinners."],
}
# What the timer's note says while the detector or the cooks work, in turn.
TASTING_LINES = ["Washing the text: hidden characters, look-alike letters…", "Dicing the sentences…",
                 "Tasting in 350-word bites…", "Simmering the paragraphs on low heat…",
                 "Sniffing for synthetic spices…", "Checking the seasoning against human recipes…",
                 "Scraping the pan for stock phrases…", "Letting the verdict rest a moment…"]
COOKING_LINES = ["Whisking the clauses…", "Folding in the writer's voice…", "Chopping long sentences into bites…",
                 "Marinating the first draft…", "Holding names and numbers aside in the fridge…",
                 "Letting the critic have a taste…", "Scraping off the burnt passages…", "Plating the paragraphs…"]
REVISING_LINES = ["Scraping off the burnt bits…", "Re-seasoning the flagged passages…", "Tasting each try…",
                  "Sending the worst bites back to the stove…", "Keeping the good parts exactly as they were…"]

STAGE_CSS = """
#od-stage { position: fixed; right: 26px; bottom: 20px; z-index: 999990; display: flex; align-items: flex-end;
  gap: 6px; pointer-events: none; }
#od-stage[hidden] { display: none; }
#od-stage .od-stage-cloche { pointer-events: auto; border: 0; padding: 0; margin: 0; background: none;
  cursor: pointer; filter: drop-shadow(0 10px 12px rgba(42,31,23,0.30)); line-height: 0; }
#od-stage .od-stage-cloche:focus-visible { outline: 2.5px solid var(--od-accent, #C2410C); outline-offset: 4px;
  border-radius: 50%; }
#od-stage .od-cl { display: block; width: 104px; height: 104px; transform-origin: 50% 85%; }
#od-stage .od-cl svg { width: 100%; height: 100%; overflow: visible; display: block; }
#od-stage .od-stage-bubble { pointer-events: auto; max-width: 240px; margin-bottom: 70px; padding: 12px 14px;
  border-radius: 16px 16px 4px 16px; font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace; font-size: 13px;
  line-height: 1.45; color: var(--od-ink, #2A1F17); background: var(--od-panel-solid, #FFFCF6);
  border: 1.5px solid var(--od-edge, #E7DAC4); box-shadow: 0 3px 0 var(--od-edge, #E7DAC4);
  transition: opacity .35s ease; }
#od-stage[data-finish="textured"] .od-stage-bubble { border-width: 1px;
  background: linear-gradient(180deg, var(--od-tx-card-top, #FFFFFF), var(--od-tx-card-bottom, #FBF3E4));
  box-shadow: 0 10px 20px -10px rgba(60,40,20,0.5); }
#od-stage .od-stage-bubble.od-in { animation: od-bub-in .28s cubic-bezier(.3,1.5,.5,1); }
#od-stage .od-stage-bubble.od-out { opacity: 0; }
#od-stage .od-stage-bubble[hidden] { display: none; }
.od-cl.cl-happy { animation: od-cl-happy .9s cubic-bezier(.3,1.6,.5,1); }
.od-cl.cl-skeptical { animation: od-cl-skep 1s ease; }
.od-cl.cl-shocked { animation: od-cl-shock .8s ease; }
.od-cl.cl-tasting { animation: od-cl-bob 1.3s ease-in-out infinite; }
@keyframes od-cl-happy { 0% { transform: scale(.8,1.15); } 35% { transform: translateY(-14px) scale(1.12,.9); }
  65% { transform: translateY(0) scale(.96,1.04); } 100% { transform: none; } }
@keyframes od-cl-skep { 0% { transform: rotate(0); } 30% { transform: rotate(-10deg); } 60% { transform: rotate(6deg); }
  100% { transform: rotate(0); } }
@keyframes od-cl-shock { 0% { transform: translateY(0); } 20% { transform: translateY(-18px); }
  35% { transform: translateY(0) translateX(-5px); } 45% { transform: translateX(5px); }
  55% { transform: translateX(-4px); } 65% { transform: translateX(3px); } 100% { transform: none; } }
@keyframes od-cl-bob { 0%, 100% { transform: translateY(0) rotate(0); }
  50% { transform: translateY(-5px) rotate(-3deg); } }
@keyframes od-bub-in { 0% { opacity: 0; transform: translateY(8px) scale(.9); } 100% { opacity: 1; transform: none; } }
@media (max-width: 640px) { #od-stage { right: 10px; bottom: 10px; } #od-stage .od-cl { width: 72px; height: 72px; }
  #od-stage .od-stage-bubble { margin-bottom: 50px; max-width: 200px; } }
@media print { #od-stage { display: none; } }
@media (prefers-reduced-motion: reduce) { .od-cl, #od-stage .od-stage-bubble, .od-kt-run .od-kt-face {
  animation: none !important; } }
"""

STAGE_JS = """
const S = window.__odStage || (window.__odStage = { session: null, seen: 0, base: "happy", face: null, poked: -1,
  svgs: {}, lines: [], bubbleTimer: null, moodTimer: null, ac: null });

function unlock() {
  try {
    S.ac = S.ac || new (window.AudioContext || window.webkitAudioContext)();
    if (S.ac.state === "suspended") S.ac.resume();
    if (!S.ring && !S.loading) {  // the recorded bell, decoded once
      S.loading = true;
      fetch(BELL_URI).then((r) => r.arrayBuffer()).then((data) => S.ac.decodeAudioData(data))
        .then((buffer) => { S.ring = buffer; }).catch(() => { S.loading = false; });
    }
  } catch (e) { /* no sound on this browser */ }
}

function bell() {
  try {
    const ac = S.ac; if (!ac) return;
    if (ac.state === "suspended") ac.resume();
    if (S.ring) {
      const source = ac.createBufferSource(), gain = ac.createGain();
      gain.gain.value = 0.9;
      source.buffer = S.ring; source.connect(gain); gain.connect(ac.destination); source.start();
      return;
    }
    synthBell();
  } catch (e) { /* no sound on this browser */ }
}

function synthBell() {  // until the recording is decoded, or where it can't be
  try {
    const ac = S.ac; if (!ac) return;
    if (ac.state === "suspended") ac.resume();
    const now = ac.currentTime;
    [[1568, 0.32], [3136, 0.12], [2349, 0.08], [4704, 0.04]].forEach(([f, v]) => {
      const o = ac.createOscillator(), g = ac.createGain();
      o.type = "sine"; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, now);
      g.gain.exponentialRampToValueAtTime(v, now + 0.006);
      g.gain.exponentialRampToValueAtTime(0.0001, now + 2.2);
      o.connect(g); g.connect(ac.destination); o.start(now); o.stop(now + 2.3);
    });
  } catch (e) { /* no sound on this browser */ }
}

function face(mood, replay) {
  const holder = document.querySelector("#od-stage .od-cl");
  if (!holder || !S.svgs[mood]) return;
  if (S.face !== mood) { holder.innerHTML = S.svgs[mood]; S.face = mood; }
  holder.setAttribute("aria-label", S.labels ? S.labels[mood] : mood);
  if (!replay && holder.classList.contains("cl-" + mood)) return;
  holder.className = "od-cl";
  void holder.offsetWidth;  // a reflow, so the same reaction can play again
  holder.classList.add("cl-" + mood);
}

function say(text, ms) {
  const bubble = document.querySelector("#od-stage .od-stage-bubble");
  if (!bubble) return;
  clearTimeout(S.bubbleTimer);
  bubble.textContent = text;
  bubble.hidden = false;
  bubble.classList.remove("od-out", "od-in");
  void bubble.offsetWidth;
  bubble.classList.add("od-in");
  S.bubbleTimer = setTimeout(() => {
    bubble.classList.add("od-out");
    S.bubbleTimer = setTimeout(() => { bubble.hidden = true; }, 400);
  }, ms || 3200);
}

function poke() {
  if (S.base === "tasting") { say(S.busyLine || "Shh. I am tasting.", 1800); return; }
  if (S.verdict && S.verdict.length) {  // after a verdict: the same face, and that verdict's lines in turn
    clearTimeout(S.moodTimer);
    face(S.base, true);
    say(S.verdict[S.said % S.verdict.length], 3800);
    S.said += 1;
    return;
  }
  if (!S.lines.length) return;
  S.poked = (S.poked + 1) % S.lines.length;
  clearTimeout(S.moodTimer);
  face("happy", true);
  say(S.lines[S.poked], 3000);
  if (S.base !== "happy") S.moodTimer = setTimeout(() => face(S.base, true), 3200);
}

function stage() {
  let root = document.getElementById("od-stage");
  if (!root) {
    root = document.createElement("div");
    root.id = "od-stage";
    root.innerHTML = '<div class="od-stage-bubble" role="status" aria-live="polite" hidden></div>'
      + '<button type="button" class="od-stage-cloche" aria-label="Cloche, the critic. Click him to say hello.">'
      + '<span class="od-cl" role="img"></span></button>';
    document.body.appendChild(root);
    root.querySelector("button").addEventListener("click", () => { unlock(); poke(); });
    // Sound needs a click on the page first; any click or key unlocks it for the bell later.
    document.addEventListener("pointerdown", unlock, { capture: true, passive: true });
    document.addEventListener("keydown", unlock, { capture: true });
  }
  if (!document.getElementById("od-stage-css")) {
    const css = document.createElement("style");
    css.id = "od-stage-css";
    css.textContent = STAGE_CSS;
    document.head.appendChild(css);
  }
  return root;
}

export default function (component) {
  const { data } = component;
  const root = stage();
  S.svgs = data.svgs || S.svgs;
  S.labels = data.labels || S.labels;
  S.lines = data.lines || [];
  S.busyLine = data.busy_line;
  root.hidden = !data.visible;
  root.dataset.finish = data.finish || "standard";
  if (data.session !== S.session) { S.session = data.session; S.seen = 0; }
  const cue = data.cue || {};
  if ((cue.id || 0) > S.seen) {
    S.seen = cue.id;
    S.base = cue.mood || "happy";
    S.verdict = cue.lines && cue.lines.length ? cue.lines : null;
    S.said = 0;
    clearTimeout(S.moodTimer);
    if (cue.mood) face(cue.mood, true);
    if (cue.say) say(cue.say, cue.ms);
    if (cue.ring) bell();
  } else if (!S.face) {
    face(S.base, false);
  }
}
""".replace("STAGE_CSS", json.dumps(STAGE_CSS)).replace(
    "BELL_URI", json.dumps("data:audio/mpeg;base64," + base64.b64encode(
        (Path(__file__).parent / "assets" / "bell.mp3").read_bytes()).decode()))

TIMER_JS = """
const TIMERS = window.__odTimers || (window.__odTimers = new Map());

function clock(seconds) {
  const s = Math.max(0, Math.ceil(seconds));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function build(host, size) {
  const ticks = Array.from({ length: 12 }, (_, i) =>
    '<i class="od-kt-tick' + (i % 3 === 0 ? " big" : "") + '" style="transform: rotate(' + (i * 30 + 180)
    + 'deg) translateY(' + (size * 0.33).toFixed(1) + 'px)"></i>').join("");
  host.innerHTML = '<div class="od-kt" role="timer" aria-live="off">'
    + '<div class="od-kt-face" style="width:' + size + 'px;height:' + size + 'px">'
    + '<i class="od-kt-winder"></i><i class="od-kt-body"></i><i class="od-kt-dial"></i>' + ticks
    + '<i class="od-kt-hub"></i></div>'
    + '<div class="od-kt-side"><span class="od-kt-eyebrow"></span><b class="od-kt-time"></b>'
    + '<span class="od-kt-note"></span></div></div>';
}

export default function (component) {
  const { data, key, parentElement } = component;
  let host = parentElement.querySelector(":scope > .od-kt-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "od-kt-host";
    parentElement.appendChild(host);
  }
  if (host.dataset.size !== String(data.size)) { build(host, data.size); host.dataset.size = String(data.size); }
  host.dataset.layout = data.layout;
  let t = TIMERS.get(key);
  const now = performance.now();
  if (!t || t.stamp !== data.stamp) {
    t = { stamp: data.stamp, endAt: null, total: null, startAt: now, steps: [] };
    TIMERS.set(key, t);
  }
  if (data.left != null) {
    const endAt = now + data.left * 1000;
    // Small changes in the estimate would make the hand stutter; a real change moves it.
    if (t.endAt === null || Math.abs(endAt - t.endAt) > 2500) t.endAt = endAt;
    t.total = Math.max(data.total || data.left, 1);
  }
  t.steps = data.steps || [];
  host.querySelector(".od-kt-eyebrow").textContent = data.eyebrow || "";
  host.querySelector(".od-kt-eyebrow").hidden = !data.eyebrow;
  const timeEl = host.querySelector(".od-kt-time"), noteEl = host.querySelector(".od-kt-note");
  const dial = host.querySelector(".od-kt-dial"), box = host.querySelector(".od-kt");
  box.classList.add("od-kt-run");
  t.lines = data.lines || [];
  const paint = (at) => {
    let label, from = 0, to, step = data.note || "";
    if (t.endAt !== null) {
      // The end eases towards a new estimate over about a second instead of jumping.
      if (t.shownEnd == null) t.shownEnd = t.endAt;
      t.shownEnd += (t.endAt - t.shownEnd) * Math.min(1, (at - (t.lastAt || at)) / 900);
      const left = (t.shownEnd - at) / 1000;
      // The red counts down a second at a time, each tick eased over about a tenth of a second.
      const target = Math.max(0, Math.min(1, Math.ceil(Math.max(0, left)) / t.total)) * 360;
      t.angle = t.angle == null ? target : t.angle + (target - t.angle) * Math.min(1, (at - (t.lastAt || at)) / 70);
      to = t.angle;
      label = left > 0 ? clock(left) : (data.over || "Any second…");
      const done = 1 - to / 360;
      for (const [share, text] of t.steps) if (done >= share) step = text;
    } else {
      // No estimate yet: a quarter of red turns smoothly round the dial.
      from = (((at - t.startAt) / 1000) * 60) % 360;
      to = 90;
      label = data.waiting || "Timing…";
    }
    t.lastAt = at;
    if (t.lines.length) step = t.lines[Math.floor((at - t.startAt) / 2600) % t.lines.length];
    dial.style.background = "conic-gradient(from " + from.toFixed(2) + "deg, var(--od-kt-red, #D7261E) 0deg "
      + to.toFixed(2) + "deg, var(--od-kt-face, #FFFCF6) " + to.toFixed(2) + "deg 360deg)";
    if (timeEl.textContent !== label) timeEl.textContent = label;
    if (noteEl.textContent !== step) {
      noteEl.textContent = step;
      noteEl.classList.remove("od-kt-new"); void noteEl.offsetWidth; noteEl.classList.add("od-kt-new");
    }
    box.setAttribute("aria-label", "Kitchen timer: " + label + (t.endAt !== null ? " left" : ""));
  };
  cancelAnimationFrame(host._frame);
  const loop = (at) => {
    if (!host.isConnected) return;
    paint(at);
    host._frame = requestAnimationFrame(loop);
  };
  host._frame = requestAnimationFrame(loop);
}
"""

TIMER_CSS = """
.od-kt-host { display: flex; justify-content: center; }
.od-kt { display: flex; align-items: center; gap: 18px; }
.od-kt-host[data-layout="stack"] .od-kt { flex-direction: column; gap: 12px; text-align: center; }
.od-kt-face { position: relative; flex: 0 0 auto; transform-origin: 50% 60%; }
.od-kt-run .od-kt-face { animation: od-kt-tick 1s ease-in-out infinite alternate; }
@keyframes od-kt-tick { 0% { transform: rotate(-0.8deg); } 100% { transform: rotate(0.8deg); } }
.od-kt-new { animation: od-kt-note .45s ease-out; }
@keyframes od-kt-note { 0% { opacity: 0; transform: translateY(4px); } 100% { opacity: 1; transform: none; } }
.od-kt-face i { position: absolute; display: block; }
.od-kt-winder { left: 50%; top: -7%; width: 22%; height: 12%; margin-left: -11%; border-radius: 6px 6px 3px 3px;
  background: linear-gradient(180deg, #E9DCC6, #BFAE93);
  box-shadow: inset 0 1px 0 #FFF7E8, 0 2px 3px rgba(42,31,23,.35); }
.od-kt-body { inset: 0; border-radius: 50%; background: linear-gradient(160deg, #FFFDF8 0%, #EFE3CF 60%, #D9C7AA 100%);
  box-shadow: 0 10px 18px -8px rgba(42,31,23,.55), inset 0 2px 0 #FFFFFF, inset 0 -4px 8px rgba(120,90,60,.25); }
.od-kt-dial { inset: 9%; border-radius: 50%; box-shadow: inset 0 3px 6px rgba(42,31,23,.35);
  border: 2px solid #CDBA9C; }
.od-kt-tick { left: 50%; top: 50%; width: 2px; height: 5%; margin-left: -1px; background: #2A1F17; opacity: .55;
  transform-origin: 50% 0; }
.od-kt-tick.big { height: 9%; }
.od-kt-hub { left: 50%; top: 50%; width: 22%; height: 22%; margin: -11% 0 0 -11%; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #FFFFFF, #D8C7AC 70%); box-shadow: 0 2px 4px rgba(42,31,23,.4); }
.od-kt-side { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.od-kt-eyebrow { font-family: var(--od-mono); font-size: .75rem; font-weight: 700; letter-spacing: .16em;
  text-transform: uppercase; color: var(--od-accent); }
.od-kt-eyebrow[hidden] { display: none; }
.od-kt-time { font-family: var(--od-mono); font-size: 1.85rem; font-weight: 800; letter-spacing: -.03em;
  color: var(--od-ink); font-variant-numeric: tabular-nums; }
.od-kt-note { font-size: .82rem; color: var(--od-muted); }
.od-kt-host[data-layout="stack"] .od-kt-note { font-size: .9rem; color: var(--od-ink); opacity: .8; }
"""

INGREDIENTS_JS = """
const ICONS = ICON_SVGS;

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let host = parentElement.querySelector(":scope > .od-ing-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "od-ing-host";
    parentElement.appendChild(host);
    host.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button || button.disabled) return;
      const points = Object.assign({}, host._points);
      const total = host._data.total, most = host._data.most;
      if (button.dataset.preset) {
        Object.assign(points, JSON.parse(button.dataset.preset));
      } else {
        const id = button.dataset.row, k = Number(button.dataset.k), n = points[id];
        const left = total - Object.values(points).reduce((a, b) => a + b, 0);
        points[id] = k === n ? k - 1 : Math.min(k, n + left, most);
      }
      if (JSON.stringify(points) === JSON.stringify(host._points)) return;
      host._pending = points;
      paint(host, points, button.dataset.row);
      setStateValue("points", points);
    });
  }
  host._data = data;
  if (host._pending && JSON.stringify(host._pending) === JSON.stringify(data.points)) host._pending = null;
  paint(host, host._pending || data.points, null);
}

function paint(host, points, popped) {
  const data = host._data;
  const spent = Object.values(points).reduce((a, b) => a + b, 0), left = data.total - spent;
  if (!host.dataset.built) {
    host.dataset.built = "1";
    host.innerHTML = '<div class="od-ing-presets" role="group" aria-label="Presets">'
      + data.presets.map((p) => '<button type="button" class="od-ing-preset" data-preset=\\''
        + JSON.stringify(p.points) + '\\'>' + p.label + '</button>').join("") + '</div>'
      + data.rows.map((r) => '<div class="od-ing-row"><span class="od-ing-label"><b>' + r.label + '</b><small>'
        + r.hint + '</small></span><span class="od-ing-items" role="group" aria-label="' + r.label + '">'
        + Array.from({ length: data.most }, (_, i) => '<button type="button" class="od-ing" data-row="' + r.id
          + '" data-k="' + (i + 1) + '" aria-label="' + r.label + ': ' + (i + 1) + ' of ' + data.most + '">'
          + ICONS[r.kind] + '</button>').join("") + '</span></div>').join("")
      + '<div class="od-ing-left"></div>';
  }
  for (const row of data.rows) {
    const n = points[row.id];
    host.querySelectorAll('button[data-row="' + row.id + '"]').forEach((b) => {
      const k = Number(b.dataset.k), on = k <= n, was = b.classList.contains("on");
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.disabled = !on && k > n + left;
      if (on && !was && popped === row.id) { b.classList.remove("pop"); void b.offsetWidth; b.classList.add("pop"); }
    });
  }
  host.querySelectorAll(".od-ing-preset").forEach((b) => {
    const on = JSON.stringify(JSON.parse(b.dataset.preset)) === JSON.stringify(
      Object.fromEntries(Object.keys(JSON.parse(b.dataset.preset)).map((k) => [k, points[k]])));
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
  host._points = points;
  const note = host.querySelector(".od-ing-left");
  note.textContent = left ? left + " of " + data.total + " left to spend: click an ingredient to add it"
    : "All " + data.total + " spent: click a lit ingredient to take it back";
  note.classList.toggle("spare", left > 0);
}
"""

# The four ingredients, 32 units square: tomato (beating the critic), basil (sounding like the writer), chilli
# (speed) and garlic (saving quota).
ICONS = {
    "tomato": '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="18" r="11.5" fill="#D7261E" '
              'stroke="#7A1410" stroke-width="1.6"/><ellipse cx="11.5" cy="14.5" rx="3" ry="2" fill="#FF8A7A" '
              'opacity="0.8"/><path d="M16 7 l2 3.5 l4 -1 l-2.6 3 l3 2.4 l-4 0 l-2.4 3 l-0.8 -3.6 l-4 -0.6 l3.4 -2 '
              'l-1.4 -3.8 l3.2 2.2 z" fill="#3F7A3A" stroke="#22471F" stroke-width="0.8" '
              'stroke-linejoin="round"/></svg>',
    "basil": '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M6 26 C 6 14, 14 5, 27 5 C 27 18, 19 27, 6 26 Z" '
             'fill="#4C9A3F" stroke="#22471F" stroke-width="1.6" stroke-linejoin="round"/><path d="M6 26 C 12 19, '
             '18 13, 25 7" fill="none" stroke="#C9E7B8" stroke-width="1.4" stroke-linecap="round"/><path d="M12 20 '
             'l-2 -4 M16 16 l-1 -5 M20 12 l0 -4 M14 18 l4 1 M18 14 l4 1" stroke="#C9E7B8" stroke-width="1" '
             'stroke-linecap="round"/></svg>',
    "chili": '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M22 7 C 26 10, 25 18, 18 23 C 13 26.5, 7 27.5, '
             '4 26 C 9 24, 13 21, 15.5 16 C 17.5 12, 18.5 8.5, 22 7 Z" fill="#E2401C" stroke="#7A1410" '
             'stroke-width="1.6" stroke-linejoin="round"/><path d="M20 12 C 19 16, 16 20, 11 23" fill="none" '
             'stroke="#FF9C7E" stroke-width="1.4" stroke-linecap="round" opacity="0.8"/><path d="M22 7 C 23 4.5, '
             '25.5 3.5, 28 4.5" fill="none" stroke="#3F7A3A" stroke-width="2.4" stroke-linecap="round"/></svg>',
    "garlic": '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 4 C 17 8, 26 11, 26 19 C 26 25, 21 28, 16 '
              '28 C 11 28, 6 25, 6 19 C 6 11, 15 8, 16 4 Z" fill="#FBF3E2" stroke="#8C7867" stroke-width="1.6" '
              'stroke-linejoin="round"/><path d="M16 9 C 13 14, 12 20, 13 27 M16 9 C 19 14, 20 20, 19 27" '
              'fill="none" stroke="#D8C7AC" stroke-width="1.3"/><path d="M14 4 l2 -2 l2 2" fill="none" '
              'stroke="#8C7867" stroke-width="1.4" stroke-linecap="round"/></svg>',
}

INGREDIENTS_CSS = """
.od-ing-host { display: flex; flex-direction: column; gap: 2px; }
.od-ing-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.od-ing-preset { height: 32px; padding: 0 14px; border-radius: 999px; border: 1px solid var(--od-edge);
  background: transparent; color: var(--od-ink); font-family: var(--od-mono); font-size: .75rem; font-weight: 500;
  cursor: pointer; transition: transform .08s; }
.od-ing-preset:hover { background: color-mix(in srgb, var(--od-edge) 45%, transparent); }
.od-ing-preset.on { background: var(--od-on); color: var(--od-on-ink); border-color: var(--od-on); font-weight: 700; }
.od-ing-preset:active { transform: translateY(2px); }
.od-ing-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 12px;
  padding: 6px 0; border-top: 1px dashed var(--od-hair); }
.od-ing-label { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.od-ing-label b { font-size: .875rem; font-weight: 500; color: var(--od-ink); }
.od-ing-label small { font-size: .7rem; color: var(--od-muted); }
.od-ing-items { display: flex; gap: 2px; }
.od-ing { width: 32px; height: 32px; padding: 2px; border: 0; background: none; cursor: pointer; border-radius: 8px;
  transition: transform .12s; }
.od-ing svg { width: 100%; height: 100%; display: block; filter: grayscale(1); opacity: .28;
  transition: filter .25s, opacity .25s; }
.od-ing.on svg { filter: none; opacity: 1; }
.od-ing:hover:not(:disabled) { transform: translateY(-2px) rotate(-6deg); }
.od-ing:disabled { cursor: default; }
.od-ing:focus-visible, .od-ing-preset:focus-visible { outline: 2.5px solid var(--od-accent); outline-offset: 2px; }
.od-ing.pop svg { animation: od-ing-pop .4s cubic-bezier(.3,1.8,.5,1); }
@keyframes od-ing-pop { 0% { transform: scale(.4) rotate(-20deg); } 100% { transform: none; } }
.od-ing-left { font-family: var(--od-mono); font-size: .72rem; color: var(--od-muted); padding-top: 6px;
  border-top: 1px dashed var(--od-hair); }
.od-ing-left.spare { color: var(--od-accent); }
@media (prefers-reduced-motion: reduce) { .od-ing.pop svg { animation: none; } }
"""
CARDS_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let host = parentElement.querySelector(":scope > .od-cards");
  if (!host) {
    host = document.createElement("div");
    host.className = "od-cards";
    host.setAttribute("role", "radiogroup");
    parentElement.appendChild(host);
    host.addEventListener("click", (event) => {
      const card = event.target.closest("button[data-id]");
      if (!card || card.getAttribute("aria-checked") === "true") return;
      host._pending = card.dataset.id;
      paint(host, card.dataset.id, true);
      setStateValue("value", card.dataset.id);
    });
  }
  host.setAttribute("aria-label", data.label);
  host.dataset.check = data.check ? "1" : "";
  const ids = data.cards.map((c) => c.id).join("|");
  if (host.dataset.ids !== ids) {
    host.dataset.ids = ids;
    host.classList.toggle("od-tiles", data.cards.some((c) => c.preview));
    host.innerHTML = data.cards.map((c, i) => '<button type="button" class="od-card" role="radio" data-id="' + c.id
      + '">' + (c.preview ? '<span class="od-card-preview" aria-hidden="true">' + c.preview + '</span>' : '')
      + '<span class="od-card-row"><span class="od-card-n" aria-hidden="true">' + (data.check ? '' : (i + 1))
      + '</span><span class="od-card-body"><b></b><small></small></span></span></button>').join("");
    data.cards.forEach((c, i) => {
      const card = host.children[i];
      card.querySelector("b").textContent = c.title;
      card.querySelector("small").textContent = c.desc;
    });
  }
  if (host._pending && host._pending === data.value) host._pending = null;
  paint(host, host._pending || data.value, false);
}

function paint(host, value, pop) {
  host.querySelectorAll("button[data-id]").forEach((card) => {
    const on = card.dataset.id === value, badge = card.querySelector(".od-card-n");
    card.setAttribute("aria-checked", on ? "true" : "false");
    if (host.dataset.check === "1") badge.textContent = on ? "✓" : "";
    card.tabIndex = on ? 0 : -1;
    if (on && pop) { badge.classList.remove("pop"); void badge.offsetWidth; badge.classList.add("pop"); }
  });
}
"""

CARDS_CSS = """
.od-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 10px; }
.od-card { display: flex; align-items: flex-start; gap: 12px; padding: 14px 15px; border-radius: 18px; text-align: left;
  color: var(--od-ink); cursor: pointer; font: inherit; border: 1px solid var(--od-edge);
  background: linear-gradient(180deg, var(--od-tx-card-top, var(--od-panel)),
    var(--od-tx-card-bottom, var(--od-panel)));
  box-shadow: inset 0 1px 0 var(--od-tx-shine, transparent), 0 2px 0 var(--od-edge);
  transition: transform .08s, border-color .15s; }
.od-card:hover { border-color: var(--od-accent); }
.od-card:active { transform: translateY(3px); }
.od-card:focus-visible { outline: 2.5px solid var(--od-accent); outline-offset: 3px; }
.od-card[aria-checked="true"] { border: 2px solid var(--od-accent); padding: 13px 14px;
  background: linear-gradient(180deg, var(--od-panel-solid), var(--od-soft, var(--od-panel-strong)));
  box-shadow: 0 4px 0 var(--od-accent), 0 14px 22px -14px var(--od-glow); }
.od-card-n { flex: 0 0 auto; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-family: var(--od-mono); font-size: .875rem; font-weight: 800; color: var(--od-ink);
  background: var(--od-track); box-shadow: inset 0 2px 3px rgba(60,40,20,.22); }
.od-card[aria-checked="true"] .od-card-n { color: #FFFCF6; background: var(--od-accent);
  box-shadow: 0 2px 0 var(--od-press, var(--od-accent)); }
.od-card-n.pop { animation: od-card-pop .4s cubic-bezier(.3,1.7,.5,1); }
@keyframes od-card-pop { 0% { transform: scale(.6); } 100% { transform: none; } }
.od-tiles .od-card { flex-direction: column; align-items: stretch; }
.od-card-row { display: flex; align-items: flex-start; gap: 12px; }
.od-card-preview { display: flex; align-items: center; justify-content: center; height: 64px; border-radius: 10px;
  overflow: hidden; }
.od-card-body { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.od-card-body b { font-family: var(--od-mono); font-size: .98rem; font-weight: 800; letter-spacing: -.01em; }
.od-card-body small { font-size: .76rem; line-height: 1.4; color: var(--od-muted); }
@media (prefers-reduced-motion: reduce) { .od-card-n.pop { animation: none; } }
"""

# The page's own styles for the timer, the ingredients and the cards, drawn in the page (ui/components.apply_theme).
CSS = TIMER_CSS + INGREDIENTS_CSS + CARDS_CSS

_components: dict[str, object] = {}


def _component(name: str, js: str):
    if name not in _components:
        from streamlit.components.v2 import component

        _components[name] = component(name, js=js, isolate_styles=False)
    return _components[name]


def enabled() -> bool:
    """Cloche, the bell and the timer belong to Bistro."""
    return style.bistro()


def cue(mood: str, say: str | None = None, ring: bool = False, ms: int | None = None,
        lines: list[str] | None = None) -> None:
    """Cloche's next reaction: his face (MOODS), a line in his speech bubble, and the kitchen bell. It plays once,
    the next time the page draws him (`show`). With `lines` (a verdict's), his face stays and a click on him says
    them in turn, starting with `say`; without, a click gives his small talk with a smile."""
    state = st.session_state.get(STATE) or {"id": 0}
    st.session_state[STATE] = {"id": state["id"] + 1, "mood": mood if mood in MOODS else "happy", "say": say,
                               "ring": ring, "ms": ms, "lines": lines or []}


def verdict_cue(verdict: str | None, probability: float | None = None, ring: bool = True,
                say: str | None = None) -> None:
    """Cloche's reaction to a verdict, with the bell."""
    import random

    mood, line = VERDICT_CUES.get(verdict or "", ("happy", "Done."))
    lines = list(VERDICT_LINES.get(verdict or "", [line]))
    first = say or random.choice(lines)
    cue(mood, first, ring=ring, ms=4200, lines=[first, *(other for other in lines if other != first)])


def greet(page: str, line: str) -> None:
    """Cloche says hello once per session, on the first page that asks."""
    if not st.session_state.get(GREETED):
        st.session_state[GREETED] = True
        cue("happy", line, ms=3800)


def show(page: str = "detect", where: str = "app") -> None:
    """Hands the latest cue to the page, for Cloche on the page's <body>. Drawn in a hidden spot in the sidebar,
    so it never takes room in the page. Call it at the end of a run, and before a long job to cue him at once."""
    from .components import CLOCHE_LABELS, cloche_svg

    session = st.session_state.setdefault("od:stage:session", uuid.uuid4().hex)
    data = {
        "session": session, "cue": st.session_state.get(STATE) or {"id": 0},
        "visible": enabled(), "finish": style.finish(),
        "svgs": {mood: cloche_svg(mood, "bistro") for mood in MOODS}, "labels": CLOCHE_LABELS,
        "lines": LINES.get(page, LINES["detect"]),
        "busy_line": "Shh. The cooks are working." if page == "lab" else "Shh. I am tasting.",
    }
    with st.sidebar, st.container(key=f"od-stage-{where}"):
        _component("od_stage", STAGE_JS)(data=data, key=f"od:cloche:{where}")


def timer(key: str, left: float | None, total: float | None = None, *, size: int = 120, layout: str = "row",
          eyebrow: str = "", note: str = "", steps: list[tuple[float, str]] | None = None, stamp: str = "",
          over: str = "Any second…", waiting: str = "Timing…", lines: list[str] | None = None) -> None:
    """The kitchen timer, counting down in the browser from `left` seconds (None: no estimate yet, the sector
    sweeps round) out of `total`. `steps` are (share of the time gone, line) pairs for the note as it goes. A new
    `stamp` starts it afresh; the same one keeps the hand where it is across reruns."""
    data = {"left": left, "total": total, "size": size, "layout": layout, "eyebrow": eyebrow, "note": note,
            "steps": steps or [], "stamp": stamp, "over": over, "waiting": waiting, "lines": lines or []}
    _component("od_kitchen_timer", TIMER_JS)(data=data, key=key)


def cards(options: list[tuple[str, str, str]], key: str, label: str, check: bool = False,
          previews: dict[str, str] | None = None) -> str:
    """Big numbered cards to pick one of (id, title, description), as the Lab's "What are we making?"; with `check`,
    a tick marks the picked one instead of numbers, and `previews` (trusted HTML from the code, by id) tops a card with
    a picture. The pick is kept in session state under `key`, so it can be set from elsewhere (a hand-off from
    Detect). Gives the pick."""
    ids = [option for option, _, _ in options]
    if st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]
    data = {"label": label, "value": st.session_state[key],
            "check": check,
            "cards": [{"id": option, "title": title, "desc": desc, "preview": (previews or {}).get(option, "")}
                      for option, title, desc in options]}
    result = _component("od_cards", CARDS_JS)(data=data, key=f"od-cards:{key}", default={"value": data["value"]},
                                              on_value_change=lambda: None)
    chosen = (result or {}).get("value") if hasattr(result, "get") else None
    seen = st.session_state.get(f"od-cards-seen:{key}")
    st.session_state[f"od-cards-seen:{key}"] = chosen
    if chosen != seen and chosen in ids:  # a click: the component sent a new value
        st.session_state[key] = chosen
    return st.session_state[key]


ROWS = (("detector", "tomato", "Beat the critic", "Tomatoes · lower AI likelihood"),
        ("style", "basil", "Sound like the recipe card", "Basil · closer style match"),
        ("speed", "chili", "Speed", "Chillies · faster, more at once"),
        ("quota", "garlic", "Save quota", "Garlic · cheaper models, less thinking"))
PRESETS = (("Balanced", {"detector": 3, "style": 3, "speed": 3, "quota": 3}),
           ("Beat the critic", {"detector": 6, "style": 3, "speed": 2, "quota": 1}),
           ("Sound like me", {"detector": 2, "style": 6, "speed": 2, "quota": 2}),
           ("Fast and cheap", {"detector": 1, "style": 2, "speed": 4, "quota": 5}))


def ingredients(points: dict[str, int], total: int, most: int, key: str = "redteam:ingredients") -> dict[str, int]:
    """The Lab's priorities as rows of ingredients, `most` of each: a lit one is a point spent. Clicking one fills
    the row up to it, or takes it back if it is the last lit one, within the `total` points. Gives the points."""
    data = {"points": points, "total": total, "most": most,
            "rows": [{"id": row, "kind": kind, "label": label, "hint": hint} for row, kind, label, hint in ROWS],
            "presets": [{"label": label, "points": chosen} for label, chosen in PRESETS
                        if sum(chosen.values()) == total and max(chosen.values()) <= most]}
    js = INGREDIENTS_JS.replace("ICON_SVGS", json.dumps(ICONS))
    result = _component("od_ingredients", js)(data=data, key=key, default={"points": points},
                                               on_points_change=lambda: None)
    chosen = (result or {}).get("points") if hasattr(result, "get") else None
    # The component keeps the last points it sent; only a new value is a click. Otherwise the session's points stand,
    # so a change made elsewhere (a reset, another page) isn't undone by what the component last said.
    seen = st.session_state.get(f"{key}:seen")
    st.session_state[f"{key}:seen"] = chosen
    if (chosen != seen and isinstance(chosen, dict) and set(chosen) == set(points)
            and all(isinstance(v, int) and 0 <= v <= most for v in chosen.values()) and sum(chosen.values()) <= total):
        return dict(chosen)
    return dict(points)
