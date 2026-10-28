"""The Detect page: paste a text or upload a document, and get a calibrated verdict, the engines'
readings and the sentences that drive the score."""

from __future__ import annotations

import hashlib

import streamlit as st

from opendetector.documents import extract_answer_slots, extract_document
from opendetector.segment import count_words

from . import models
from .components import (
    VERDICTS,
    Reading,
    contribution_chart,
    engine_panel,
    highlighted_text,
    keep,
    persistent_upload,
    score_bars,
    sentence_chart,
    verdict_badge,
    verdict_card,
)

DOCUMENT_TYPES = ("md", "docx", "pptx")
SOURCES = ("Paste text", "Upload a file")
HIGHLIGHTS = ("Every sentence", "Flagged only")
MIN_WORDS = 150  # DetectorConfig.min_words, for the hint under the input
RESULT = "detect:result"
# Set by the Red team page to send a sample here: the text goes in the paste box and is analyzed.
AUTORUN = "detect:autorun"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _settings(profile: str) -> tuple[list[str], bool]:
    """The page's sidebar: which engines to run and whether to score sentences."""
    available = models.engines_of(profile)
    with st.sidebar:
        st.subheader("Analysis")
        enabled = available
        if len(available) > 1:
            enabled = [
                name for name in available
                if st.toggle(models.ENGINES[name][0], key=keep(f"keep:detect:engine:{name}", True),
                             help=models.ENGINES[name][1])
            ]
            if "bino" in enabled and models.RAM_GB < 12:
                st.caption("Binoculars takes about 2 GB of the 3.5 GB. If this computer slows down, switch "
                           "it off.")
        sentences = st.toggle("Sentence highlights", key=keep("keep:detect:sentences", True),
                              help="Scores ~100-word windows inside each paragraph to show where the signal "
                                   "is. Takes a little longer.")
    return enabled, sentences


def _reference_material() -> None:
    """Documents to read alongside the one being scored. Their text is never scored."""
    with st.expander("Reference material", icon=":material/menu_book:"):
        files = st.file_uploader("Markdown, Word or PowerPoint files to read alongside. They are never scored.",
                                 type=DOCUMENT_TYPES, accept_multiple_files=True, key="detect:references")
        for file in files or []:
            try:
                sections = extract_document(file.name, file.getvalue())
            except (ImportError, ValueError) as exc:
                st.error(f"Could not read {file.name}: {exc}")
                continue
            for section in sections:
                st.markdown(f"**{file.name} · {section.label}**")
                st.text(section.text)


def _input() -> tuple[str, str, list | None]:
    """The text to score, where it came from, and its sections if it is a document."""
    source = st.segmented_control("Input", SOURCES, key=keep("keep:detect:source", SOURCES[0]), required=True,
                                  label_visibility="collapsed")
    if source == SOURCES[0]:
        text = st.text_area("Text to check", key=keep("keep:detect:text", ""), height=300,
                            label_visibility="collapsed",
                            placeholder="Paste an essay, a report or a draft. At least 150 words of prose gets "
                                        "a verdict.")
        return text, "Pasted text", None

    upload = persistent_upload(
        "Markdown, Word or PowerPoint file", "detect:file", DOCUMENT_TYPES,
        help="The whole document gets a verdict, and every section with 150 words or more gets its own.",
    )
    _reference_material()
    if not upload:
        return "", "", None
    try:
        sections = extract_document(upload.name, upload.data)
        slots = extract_answer_slots(upload.name, upload.data)
    except (ImportError, ValueError) as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return "", upload.name, None
    answered = [slot.target for slot in slots if slot.previous_answer]
    if answered and st.toggle(
        f"Score only the {len(answered)} filled-in answer cells", key=keep("keep:detect:answers_only", False),
        help="Worksheets are mostly short table cells, and their prompts aren't the writer's. This scores "
             "just the right-hand cells of question/answer tables.",
    ):
        sections = answered
    if not sections:
        st.warning(f"Found no text in {upload.name}.")
        return "", upload.name, None
    with st.expander(f"Extracted text · {len(sections)} sections", icon=":material/article:"):
        for section in sections:
            st.markdown(f"**{section.label}**")
            st.text(section.text)
    return "\n\n".join(section.text for section in sections), upload.name, sections


def _analyze(text: str, source: str, sections: list | None, profile: str, enabled: list[str],
             sentences: bool) -> None:
    if not models.is_loaded(profile, enabled):
        names = ", ".join(models.ENGINES[name][0] for name in enabled)
        with st.spinner(f"Loading {names}. The first run downloads the models…"):
            detector = models.get_detector(profile, enabled)
    else:
        detector = models.get_detector(profile, enabled)
    with st.spinner("Scoring the text with every engine…"):
        report = detector.analyze(text, sentences=sentences)

    rows = []
    if sections:
        progress = st.progress(0.0, text="Scoring sections…")
        for done, section in enumerate(sections, 1):
            words = count_words(section.text)
            verdict = probability = None
            if words >= detector.config.min_words:
                section_report = detector.analyze(section.text, sentences=False)
                verdict, probability = section_report.verdict, section_report.ai_probability
            rows.append({"label": section.label, "text": section.text, "words": words, "verdict": verdict,
                         "probability": probability})
            progress.progress(done / len(sections), text=f"Scored {done} of {len(sections)} sections")
        progress.empty()

    st.session_state[RESULT] = {
        "report": report,
        "bands": [detector.verdict(s.ai_probability) for s in report.sentences],
        "sections": rows,
        "source": source,
        "profile": profile,
        "engines": enabled,
        "digest": _digest(text),
        "missing": detector.missing_features,
        "info": detector.combiner.info if detector.combiner else None,
        "contributions": (detector.combiner.contributions(report.features)
                          if detector.combiner and report.features else None),
        "intercept": detector.combiner.intercept if detector.combiner else None,
        "min_words": detector.config.min_words,
    }


def _engine_readings(report, bands: list[str]) -> None:
    readings = []
    for name, value in report.engines.items():
        label, description = models.ENGINES[name]
        if name == "bino":
            readings.append(Reading(label, f"{value:.2f}", f"{description}. A perplexity ratio: lower is more "
                                                            "AI-like."))
        else:
            readings.append(Reading(label, f"{value:.0%}", f"{description}. Its own P(AI), before calibration.",
                                    value))
    stats = [(f"{report.words:,}", "words")]
    if report.sentences:
        words = [count_words(s.text) for s in report.sentences]
        flagged = sum(w for w, band in zip(words, bands, strict=True) if band in ("likely_ai", "needs_review"))
        stats += [(f"{len(report.sentences):,}", "sentences"), (f"{flagged / max(sum(words), 1):.0%}", "words flagged")]
    engine_panel(readings, stats)


def _sections_table(rows: list[dict], min_words: int, thresholds: dict[str, float]) -> None:
    st.caption(f"Each section with {min_words} words or more is scored on its own. Shorter ones are too short "
               f"for a reliable verdict. The dashed lines are the cut-offs at {thresholds['review']:.0%} and "
               f"{thresholds['ai']:.0%}.")
    if chart := score_bars(rows, thresholds):
        st.altair_chart(chart, width="stretch")
    st.dataframe(
        [{"Section": row["label"], "Words": row["words"],
          "Verdict": VERDICTS[row["verdict"]][0] if row["verdict"] else "Too short",
          "AI likelihood": row["probability"]} for row in rows],
        hide_index=True, width="stretch",
        column_config={"AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
    )
    labels = [row["label"] for row in rows]
    chosen = st.selectbox("Read a section", labels, index=None, placeholder="Choose a section to read")
    if chosen:
        row = rows[labels.index(chosen)]
        with st.container(border=True):
            verdict_badge(row["verdict"] or "insufficient_text")
            st.text(row["text"])


def _engine_pushes(result: dict) -> None:
    """What moved the score: each engine's terms in the combiner's log-odds, when there are two or more
    engines to compare."""
    terms, intercept = result.get("contributions"), result.get("intercept")
    if not terms or intercept is None:
        return
    pushes: dict[str, float] = {}
    for feature, value in terms.items():
        name = feature.split("_")[0]
        label = models.ENGINES[name][0] + ("" if name in result["engines"] else " (off)")
        pushes[label] = pushes.get(label, 0.0) + value
    if len(pushes) < 2:
        return
    st.markdown("**What moved the score**")
    st.altair_chart(contribution_chart(pushes), width="stretch")
    total = intercept + sum(pushes.values())
    off = " An engine that is off counts at a typical value." if any("(off)" in label for label in pushes) else ""
    st.caption(f"The combiner starts at {intercept:+.2f} and adds each engine's push, for {total:+.2f} in all. "
               f"The AI likelihood is that total through a logistic curve: {result['report'].ai_probability:.1%}."
               + off)


def _details(result: dict) -> None:
    report = result["report"]
    if report.ai_probability is not None:
        _engine_pushes(result)
    info = result["info"]
    build = models.PROFILE_LABELS[result["profile"]]
    engines = ", ".join(models.ENGINES[name][0] for name in result["engines"])
    st.markdown(f"**Build:** {build} · **Engines:** {engines or 'none'}")
    if info:
        st.markdown(
            f"**Calibration:** a logistic-regression combiner fitted on {info.get('n_fit', '?')} texts "
            f"({', '.join(info.get('datasets', [])) or 'unknown data'}), with cut-offs set on "
            f"{info.get('n_calibration_human', '?')} held-out human texts."
        )
        if "test_tpr_at_1pct_fpr" in info:
            st.markdown(f"**Held-out test:** catches {info['test_tpr_at_1pct_fpr']:.1%} of AI texts at a 1% "
                        f"false-positive rate (AUROC {info.get('test_auroc', float('nan')):.3f}).")
    else:
        st.markdown("**Calibration:** none. The score is a plain average of the classifiers.")
    thresholds = report.thresholds
    st.markdown(f"**Cut-offs:** needs review above {thresholds['review']:.1%}, likely AI above {thresholds['ai']:.1%}.")
    if report.features:
        st.dataframe([{"Feature": name, "Value": round(value, 4)} for name, value in report.features.items()],
                     hide_index=True)
    norm = report.normalization
    st.caption(f"Normalization removed {norm.invisible_chars} hidden characters and {norm.homoglyphs} look-alike "
               f"letters, and folded {norm.compatibility_chars} compatibility characters.")


def _results(result: dict, digest: str) -> None:
    report = result["report"]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Result", anchor=False)
        st.badge(result["source"], icon=":material/description:", color="gray")
        st.badge(f"{models.PROFILE_LABELS[result['profile']]} build", icon=":material/memory:", color="gray")
    if result["digest"] != digest:
        st.caption(":material/info: The input has changed since this analysis. Analyze again to update it.")

    norm = report.normalization
    if norm.suspicious:
        found = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in
                 ((norm.invisible_chars, "hidden character"), (norm.homoglyphs, "look-alike letter")) if n]
        st.error(f"Found {' and '.join(found)}, such as zero-width spaces or a Cyrillic “а” in place of a Latin "
                 "“a”. They change nothing a reader sees and are a common trick for fooling detectors. They were "
                 "removed before scoring.", icon=":material/warning:")
    if result["missing"]:
        st.warning(f"The combiner was fitted with engines that are switched off ({', '.join(result['missing'])}). "
                   "Their inputs are filled with typical values, which weakens the score.", icon=":material/tune:")

    if report.verdict == "insufficient_text":
        st.info(f"Need at least {result['min_words']} words of prose for a verdict (got {report.words}). Short "
                "texts give confident but unreliable scores.", icon=":material/short_text:")
    elif report.verdict == "not_prose":
        st.info("This looks like poetry, a list or another line-by-line format. The detector is built for prose "
                "and gives unreliable scores on these, so it doesn't score them (Turnitin skips poetry, scripts "
                "and lists too).", icon=":material/format_list_bulleted:")
    else:
        left, right = st.columns([3, 2], gap="medium", vertical_alignment="center")
        with left:
            foot = (f"{report.words:,} words · " + ("calibrated" if report.calibrated else "uncalibrated")
                    + f" · cut-offs at {report.thresholds['review']:.0%} and {report.thresholds['ai']:.0%}")
            verdict_card(report.verdict, report.ai_probability, report.thresholds, report.calibrated, foot)
        with right:
            _engine_readings(report, result["bands"])

    names = (["Highlights"] if report.ai_probability is not None else []) \
        + (["Sentences"] if report.sentences else []) + (["Sections"] if result["sections"] else []) + ["Details"]
    for tab, name in zip(st.tabs(names), names, strict=True):
        with tab:
            if name == "Highlights":
                if report.sentences:
                    mode = st.segmented_control("Highlight", HIGHLIGHTS, key=keep("keep:detect:highlight",
                                                                                  HIGHLIGHTS[0]),
                                                required=True, label_visibility="collapsed")
                    st.html(highlighted_text(report.text, report.sentences, result["bands"], report.thresholds,
                                             flagged_only=mode == HIGHLIGHTS[1]),
                            unsafe_allow_javascript=True)
                    st.caption("Stronger shades sit deeper in their band. Likely AI is underlined, Needs review "
                               "dotted. Each sentence takes the average score of the ~100-word windows around "
                               "it, so sentences next to a strongly AI-like passage get pulled toward it. "
                               "Highlights show where to look; the document score is the number to go by.")
                else:
                    st.caption("Sentence highlights are off. Switch them on in the sidebar and analyze again.")
            elif name == "Sentences":
                st.caption("Each bar is one sentence, as wide as its word count, in reading order. The dashed "
                           "lines are the document's cut-offs; sentence scores aren't calibrated on their own.")
                st.altair_chart(sentence_chart(report.sentences, result["bands"], report.thresholds),
                                width="stretch")
                st.dataframe(
                    [{"#": i, "AI likelihood": s.ai_probability, "Band": VERDICTS[band][0], "Sentence": s.text}
                     for i, (s, band) in enumerate(zip(report.sentences, result["bands"], strict=True), 1)],
                    hide_index=True, width="stretch",
                    column_config={"AI likelihood": st.column_config.ProgressColumn(
                        format="percent", min_value=0, max_value=1, width="small"),
                        "Band": st.column_config.TextColumn(width="small")},
                )
            elif name == "Sections":
                _sections_table(result["sections"], result["min_words"], report.thresholds)
            else:
                _details(result)

    st.caption(":material/balance: A detector score is statistical evidence, not proof of authorship. Don't use "
               "it as the only basis for an academic integrity decision.")


def render() -> None:
    profile = models.active_profile()
    enabled, sentences = _settings(profile)

    st.title("Detect AI-generated text", anchor=False)
    if profile == "slim":
        st.markdown("The Desklib classifier scores overlapping windows of the whole text, and a calibrated model "
                    "turns those scores into one AI likelihood.")
    else:
        st.markdown("Two fine-tuned classifiers and the Binoculars zero-shot detector score overlapping windows of "
                    "the whole text, and a calibrated model combines them into one AI likelihood.")

    with st.container(border=True):
        text, source, sections = _input()
        words = count_words(text) if text.strip() else 0
        with st.container(horizontal=True, vertical_alignment="center"):
            clicked = st.button("Analyze", type="primary", icon=":material/search:")
            if words:
                st.caption(f"{words:,} words" + ("" if words >= MIN_WORDS else
                                                 f" · a verdict needs at least {MIN_WORDS}"))

    if clicked or st.session_state.pop(AUTORUN, False):
        if not text.strip():
            st.warning("Paste some text or upload a file first.")
        elif not enabled:
            st.warning("Switch on at least one engine in the sidebar.")
        else:
            _analyze(text, source, sections, profile, enabled, sentences)

    if result := st.session_state.get(RESULT):
        _results(result, _digest(text))
