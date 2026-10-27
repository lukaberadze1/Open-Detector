"""Streamlit front end: streamlit run app.py

Set OPENDETECTOR_PROFILE=slim to run Desklib alone on the CPU, for small hosts.
"""

from __future__ import annotations

import html
import os

import streamlit as st

from opendetector import Detector, DetectorConfig
from opendetector.combiner import Combiner
from opendetector.documents import extract_document
from opendetector.segment import count_words

st.set_page_config(page_title="Open Detector", page_icon="🔎", layout="wide")

VERDICT_LABELS = {"likely_ai": "Likely AI", "needs_review": "Needs review", "likely_human": "Likely human"}
HIGHLIGHT = {"likely_ai": "rgba(255, 82, 82, 0.35)", "needs_review": "rgba(255, 193, 7, 0.35)"}
ENGINE_LABELS = {
    "glyph": "GLYPH · DeBERTa-v3-base classifier",
    "desklib": "Desklib · DeBERTa-v3-large classifier",
    "bino": "Binoculars · zero-shot (two small LMs)",
}

PROFILE = os.environ.get("OPENDETECTOR_PROFILE", "full")
CONFIG = DetectorConfig.for_profile(PROFILE)
AVAILABLE = [name for name, model in (("glyph", CONFIG.glyph_model), ("desklib", CONFIG.desklib_model),
                                      ("bino", CONFIG.binoculars_observer)) if model]

with st.sidebar:
    st.header("Engines")
    enabled = {name: st.checkbox(ENGINE_LABELS[name], value=True) for name in AVAILABLE}
    show_sentences = st.checkbox("Sentence highlights", value=True)
    if PROFILE == "slim":
        st.caption("Slim build: Desklib alone, the strongest single engine, on CPU to fit a small free host.")
    else:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        if ram_gb < 12:
            st.caption(f"This machine has {ram_gb:.0f} GB of RAM. All three engines need about 3.5 GB; "
                       "if it slows down, switch off Binoculars (about 2 GB).")


# One cache entry per engine, so toggling engines never loads a second copy of a model.
@st.cache_resource(show_spinner="Loading model (the first run downloads it)…")
def load_engine(name: str):
    from opendetector.engines import build_engine

    return build_engine(name, CONFIG)


classifiers = [load_engine(n) for n in ("glyph", "desklib") if enabled.get(n)]
combiner = Combiner.load(CONFIG.combiner_path) if CONFIG.combiner_path.exists() else None
detector = Detector(classifiers, load_engine("bino") if enabled.get("bino") else None, combiner, CONFIG)

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
if PROFILE == "slim":
    st.markdown("The Desklib classifier scores overlapping windows of the whole text, and a calibrated "
                "model turns those scores into one AI likelihood.")
else:
    st.markdown("An ensemble of two fine-tuned classifiers and the Binoculars zero-shot detector, "
                "scored over overlapping windows and combined by a calibrated model.")

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

input_mode = st.radio("Input", ("Paste text", "Upload Word or PowerPoint"), horizontal=True)
uploaded_sections = []
if input_mode == "Paste text":
    text_input = st.text_area("Paste the text to check:", height=260, placeholder="Paste an essay or draft…")
else:
    reference_uploads = st.file_uploader(
        "Optional reference documents (Word or PowerPoint)",
        type=("docx", "pptx"),
        accept_multiple_files=True,
        key="reference_uploads",
        help="Reference files are extracted for review and kept separate from the document being scored.",
    )
    reference_sections = []
    for reference in reference_uploads:
        try:
            reference_sections.extend(
                (reference.name, section)
                for section in extract_document(reference.name, reference.getvalue())
            )
        except (ImportError, ValueError) as exc:
            st.error(f"Could not read {reference.name}: {exc}")
    if reference_sections:
        with st.expander(f"Review extracted reference material ({len(reference_sections)} sections)"):
            for filename, section in reference_sections:
                st.markdown(f"**{filename} · {section.label}**")
                st.text(section.text)
        st.caption("Reference material is not included in the scored document.")

    uploaded = st.file_uploader("Choose a .docx or .pptx file", type=("docx", "pptx"))
    text_input = ""
    if uploaded:
        try:
            uploaded_sections = extract_document(uploaded.name, uploaded.getvalue())
            text_input = "\n\n".join(section.text for section in uploaded_sections)
            st.caption(f"Extracted {len(uploaded_sections)} sections from {uploaded.name}.")
            with st.expander("Review extracted text by source location"):
                for section in uploaded_sections:
                    st.markdown(f"**{section.label}**")
                    st.text(section.text)
        except (ImportError, ValueError) as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Paste some text first.")
        st.stop()

    with st.spinner("Scoring windows with every engine…"):
        report = detector.analyze(text_input, sentences=show_sentences)

    if uploaded_sections:
        st.subheader("Document sections")
        st.caption(
            "The full document gets its own calibrated verdict. Sections with enough prose are "
            "scored separately; short sections are shown without a verdict. Scores are statistical evidence, not proof."
        )
        section_results = []
        for section in uploaded_sections:
            words = count_words(section.text)
            if words < detector.config.min_words:
                section_results.append((section, words, None))
            else:
                with st.spinner(f"Scoring {section.label}…"):
                    section_results.append((section, words, detector.analyze(section.text, sentences=False)))
        for section, words, section_report in section_results:
            title = section.label
            if section_report and section_report.verdict in VERDICT_LABELS:
                title += f" · {VERDICT_LABELS[section_report.verdict]} · {section_report.ai_probability:.0%}"
            elif section_report:
                title += f" · {section_report.verdict.replace('_', ' ').title()}"
            else:
                title += f" · Too short ({words} words)"
            with st.expander(title):
                st.write(section.text)
                if section_report and section_report.verdict == "insufficient_text":
                    st.caption(f"Need at least {detector.config.min_words} words for a verdict.")
                elif section_report and section_report.verdict == "not_prose":
                    st.caption("This section is not prose, so the detector abstains.")

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
    if report.verdict == "not_prose":
        st.info(
            "This looks like poetry, a list or another line-by-line format. The detector is built "
            "for prose and gives unreliable scores on these, so it doesn't score them (Turnitin "
            "skips poetry, scripts and lists too)."
        )
        st.stop()

    cols = st.columns(2 + len(report.engines))
    cols[0].metric("Verdict", VERDICT_LABELS[report.verdict])
    cols[1].metric("AI likelihood" + ("" if report.calibrated else " (uncalibrated)"), f"{report.ai_probability:.0%}")
    names = {"glyph": "GLYPH P(AI)", "desklib": "Desklib P(AI)", "bino": "Binoculars"}
    for col, (name, value) in zip(cols[2:], report.engines.items(), strict=True):
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
                style = f"background:{HIGHLIGHT[band]};border-radius:3px;padding:1px 2px"
                body = f"<span title='{tip}' style='{style}'>{body}</span>"
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
