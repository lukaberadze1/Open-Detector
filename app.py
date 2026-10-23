"""Streamlit front end: streamlit run app.py"""

from __future__ import annotations

import html
import os

import streamlit as st

from opendetector import Detector, DetectorConfig
from opendetector.combiner import Combiner

st.set_page_config(page_title="Open Detector", page_icon="🔎", layout="wide")

VERDICT_LABELS = {"likely_ai": "Likely AI", "needs_review": "Needs review", "likely_human": "Likely human"}
HIGHLIGHT = {"likely_ai": "rgba(255, 82, 82, 0.35)", "needs_review": "rgba(255, 193, 7, 0.35)"}

with st.sidebar:
    st.header("Engines")
    use_glyph = st.checkbox("GLYPH · DeBERTa-v3-base classifier", value=True)
    use_desklib = st.checkbox("Desklib · DeBERTa-v3-large classifier", value=True)
    use_bino = st.checkbox("Binoculars · zero-shot (two small LMs)", value=True)
    show_sentences = st.checkbox("Sentence highlights", value=True)
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    if ram_gb < 12:
        st.caption(f"This machine has {ram_gb:.0f} GB of RAM. All three engines need about 3.5 GB; "
                   "if it slows down, switch off Binoculars (about 2 GB).")


CONFIG = DetectorConfig()


# One cache entry per engine, so toggling engines never loads a second copy of a model.
@st.cache_resource(show_spinner="Loading model (the first run downloads it)…")
def load_engine(name: str):
    from opendetector.engines import build_engine

    return build_engine(name, CONFIG)


classifiers = [load_engine(n) for n, on in (("glyph", use_glyph), ("desklib", use_desklib)) if on]
combiner = Combiner.load(CONFIG.combiner_path) if CONFIG.combiner_path.exists() else None
detector = Detector(classifiers, load_engine("bino") if use_bino else None, combiner, CONFIG)

with st.sidebar:
    st.header("Calibration")
    if detector.combiner is None:
        st.caption("No combiner fitted. Scores are a plain average of the classifiers.")
    else:
        info = detector.combiner.info
        st.caption(
            f"Combiner fitted on {info.get('n_fit', '?')} texts; thresholds set on "
            f"{info.get('n_calibration_human', '?')} held-out human texts."
        )
        if "test_tpr_at_1pct_fpr" in info:
            st.caption(f"Held-out test: catches {info['test_tpr_at_1pct_fpr']:.1%} of AI texts at 1% FPR.")

st.title("Open Detector")
st.markdown(
    "An ensemble of two fine-tuned classifiers and the Binoculars zero-shot detector, "
    "scored over overlapping windows and combined by a calibrated model."
)

if detector.combiner is None:
    st.warning(
        "**Uncalibrated.** No combiner has been fitted, so the score is a raw average and the "
        "verdict cut-offs are placeholders. Run the eval harness (see DETECTOR.md) to fit one."
    )
elif detector.missing_features:
    st.warning(
        f"The combiner was fitted with engines that are switched off ({', '.join(detector.missing_features)}). "
        "Their inputs are filled with typical values, which weakens the score."
    )

text_input = st.text_area("Paste the text to check:", height=260, placeholder="Paste an essay or draft…")

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Paste some text first.")
        st.stop()

    with st.spinner("Scoring windows with every engine…"):
        report = detector.analyze(text_input, sentences=show_sentences)

    norm = report.normalization
    if norm.suspicious:
        found = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in
                 ((norm.invisible_chars, "hidden character"), (norm.homoglyphs, "look-alike letter")) if n]
        st.error(
            f"Found {' and '.join(found)} (e.g. zero-width spaces, or Cyrillic “а” in place of Latin “a”). "
            "They change nothing a reader sees and are a common trick for fooling detectors. "
            "They were removed before scoring."
        )

    if report.verdict == "insufficient_text":
        st.info(
            f"Need at least {detector.config.min_words} words of prose for a verdict "
            f"(got {report.words}). Short texts give confident but unreliable scores."
        )
        st.stop()

    cols = st.columns(2 + len(report.engines))
    cols[0].metric("Verdict", VERDICT_LABELS[report.verdict])
    cols[1].metric("AI likelihood" + ("" if report.calibrated else " (uncalibrated)"), f"{report.ai_probability:.0%}")
    names = {"glyph": "GLYPH P(AI)", "desklib": "Desklib P(AI)", "bino": "Binoculars"}
    for col, (name, value) in zip(cols[2:], report.engines.items()):
        col.metric(names.get(name, name), f"{value:.2f}" if name == "bino" else f"{value:.0%}",
                   help="Lower means more AI-like." if name == "bino" else None)

    if report.sentences:
        st.subheader("Where the signal is")
        st.caption(
            "Each sentence takes the average score of the ~100-word windows around it, so sentences next "
            "to a strongly AI-like passage get pulled toward it. Highlights show where to look; the "
            "document score above is the number to go by."
        )
        parts, cursor = [], 0
        for s in report.sentences:
            parts.append(html.escape(report.text[cursor : s.start]).replace("\n", "<br>"))
            band = detector.verdict(s.ai_probability)
            body = html.escape(s.text)
            if band in HIGHLIGHT:
                tip = html.escape(f"{s.ai_probability:.0%} AI likelihood", quote=True)
                body = f"<span title='{tip}' style='background:{HIGHLIGHT[band]};border-radius:3px;padding:1px 2px'>{body}</span>"
            parts.append(body)
            cursor = s.end
        st.markdown(f"<div style='line-height:1.8;font-size:16px'>{''.join(parts)}</div>", unsafe_allow_html=True)

        with st.expander("Sentence scores"):
            st.dataframe(
                [{"#": i, "AI likelihood": f"{s.ai_probability:.0%}", "sentence": s.text}
                 for i, s in enumerate(report.sentences, 1)],
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "A detector score is statistical evidence, not proof of authorship. Don't use it as the "
        "only basis for an academic integrity decision."
    )
