"""The Detect page: paste a text or upload a document, and get a calibrated verdict, the engines'
readings and the sentences that drive the score."""

from __future__ import annotations

import hashlib
import html
import os
import time

import streamlit as st

from opendetector.documents import extract_answer_slots, extract_document
from opendetector.forensics import inspect_document
from opendetector.segment import count_words, group_parts, sentence_runs, split_sentences

from . import models, nav, style
from .components import (
    VERDICTS,
    Reading,
    contribution_chart,
    engine_panel,
    highlighted_text,
    keep,
    persistent_upload,
    pickable_highlights,
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
# Set by the Adversarial Lab to send a labelled challenge sample here for detector analysis.
AUTORUN = "detect:autorun"
PENDING = "detect:pending_text"  # text to put in Source and analyze at the top of the next run (a revise, an undo)
REVISED = "detect:revised"  # what the last revise did, with the text before it for Undo
PICK_ROUND = "detect:pick_round"  # bumped to start the picked sentences afresh


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _engines(profile: str) -> tuple[list[str], bool]:
    """The Engines panel, right above Analyze: the build, which of its engines run, and whether to score
    sentences. The build is shared with the Adversarial Lab."""
    with style.group("Engines", "02"):
        models.profile_picker()
        available = models.engines_of(profile)
        enabled = available
        if len(available) > 1:
            with st.container(horizontal=True, gap="medium"):
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


def _load(profile: str, enabled: list[str]):
    if models.is_loaded(profile, enabled):
        return models.get_detector(profile, enabled)
    names = ", ".join(models.ENGINES[name][0] for name in enabled)
    with st.spinner(f"Loading {names}. The first time, this downloads about "
                    f"{models.download_gb(profile, enabled):.1f} GB and can take a few minutes…"):
        return models.get_detector(profile, enabled)


def _analyze(text: str, source: str, sections: list | None, profile: str, enabled: list[str],
             sentences: bool) -> None:
    detector = _load(profile, enabled)
    with st.spinner("Scoring the text with every engine…"):
        report = detector.analyze(text, sentences=sentences)

    if sections:
        parts = [(section.label, section.text) for section in sections]
    else:  # pasted text: its paragraphs, short ones together and long ones cut, about 150 words a part
        clean = report.text
        found = split_sentences(clean)
        runs = sentence_runs(found, detector.config.min_words)
        parts = [(f"Part {n}", clean[found[first].start:found[last].end]) for n, (first, last) in enumerate(runs, 1)]
        parts = parts if len(parts) > 1 else []
    rows = score_parts(detector, parts)

    result = _result(detector, report, text, source, profile, enabled, rows)
    upload = st.session_state.get("upload:detect:file") if sections else None
    result["forensics"] = inspect_document(upload.name, upload.data) if upload else None
    st.session_state[RESULT] = result


def score_parts(detector, parts: list[tuple[str, str]], scored: list[dict] | None = None) -> list[dict]:
    """A row for each group of neighbouring parts (label, text) long enough to score: a part of the detector's
    minimum or more on its own, shorter ones together (group_parts), so none is left as too short unless the whole
    text is. `scored` are rows already scored for the same parts, kept for a part that stays on its own."""
    min_words = detector.config.min_words
    groups = group_parts([count_words(text) for _, text in parts], min_words)
    rows = []
    progress = st.progress(0.0, text="Scoring sections…") if len(groups) > 1 else None
    for done, group in enumerate(groups, 1):
        labels = [parts[i][0] for i in group]
        text = "\n\n".join(parts[i][1] for i in group)
        words = count_words(text)
        if len(group) == 1 and scored and scored[group[0]].get("verdict") not in (None, "insufficient_text"):
            rows.append(scored[group[0]])
            continue
        verdict = probability = None
        if words >= min_words:
            part_report = detector.analyze(text, sentences=False)
            verdict, probability = part_report.verdict, part_report.ai_probability
        label = labels[0] if len(labels) == 1 else f"{labels[0]} – {labels[-1]} ({len(labels)} parts)"
        rows.append({"label": label, "text": text, "words": words, "verdict": verdict, "probability": probability,
                     "parts": len(group)})
        if progress:
            progress.progress(done / len(groups), text=f"Scored {done} of {len(groups)} sections")
    if progress:
        progress.empty()
    return rows


def breakdown(text: str, source: str, profile: str, rows: list[dict]) -> dict:
    """Everything show_result draws for a text scored elsewhere, by every engine of `profile`. `rows` are
    its parts, already scored: label, text, words, verdict and probability."""
    enabled = models.engines_of(profile)
    detector = _load(profile, enabled)
    with st.spinner("Scoring every sentence…"):
        report = detector.analyze(text, sentences=True)
    if any(row["words"] < detector.config.min_words for row in rows):  # short blocks rated together, not "too short"
        rows = score_parts(detector, [(row["label"], row["text"]) for row in rows], rows)
    return _result(detector, report, text, source, profile, enabled, rows)


def _result(detector, report, text: str, source: str, profile: str, enabled: list[str], rows: list[dict]) -> dict:
    return {
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
    st.caption(f"Each section with {min_words} words or more is scored on its own, and shorter ones together with "
               f"their neighbours until they reach {min_words}, the fewest the detector rates reliably. The dashed "
               f"lines are the cut-offs at {thresholds['review']:.0%} and {thresholds['ai']:.0%}.")
    if chart := score_bars(rows, thresholds):
        st.altair_chart(chart, width="stretch")
    st.dataframe(
        [{"Section": row["label"], "Words": row["words"],
          "Verdict": VERDICTS[row["verdict"]][0] if row["verdict"] else "Too short",
          "AI likelihood": row["probability"]} for row in rows],
        hide_index=True, width="stretch",
        column_config={"AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                                        color=style.tokens()["accent"])},
    )
    chosen = st.selectbox("Read a section", range(len(rows)), index=None, format_func=lambda i: rows[i]["label"],
                          placeholder="Choose a section to read")  # by position: two slides' boxes can share a label
    if chosen is not None:
        row = rows[chosen]
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


def _file_check(forensics) -> None:
    """What the file says about itself. None of it counts toward the score."""
    st.caption("From the file itself, not its words. None of this changes the score. Red items are text a reader "
               "can't see; the others have ordinary explanations too, so treat them as places to ask, not proof.")
    for finding in forensics.findings:
        if finding.level == "flag":
            st.error(f"**{finding.title}.** {finding.detail}", icon=":material/warning:")
        else:
            st.info(f"**{finding.title}.** {finding.detail}", icon=":material/search:")
    if not forensics.findings:
        st.success("Nothing unusual in this file.", icon=":material/check:")
    st.dataframe([{"Property": key, "Value": str(value)} for key, value in forensics.properties.items()],
                 hide_index=True, width="stretch")


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
        used = result.get("contributions") or {}
        st.dataframe([{"Feature": name, "Value": round(value, 4), "In the score": name in used}
                      for name, value in report.features.items()], hide_index=True)
        st.caption("burst: how much sentence lengths vary. fast: Fast-DetectGPT, higher is more AI-like. perp: raw "
                   "perplexity, lower is more AI-like, but it also runs low on plain writing. zippy: how well the "
                   "text compresses after a sample of AI text, higher is more AI-like. Only the features "
                   "marked as in the score change the result.")
    norm = report.normalization
    st.caption(f"Normalization removed {norm.invisible_chars} hidden characters and {norm.homoglyphs} look-alike "
               f"letters, and folded {norm.compatibility_chars} compatibility characters.")


def _revising_offered() -> bool:
    """Revising is a drafting aid: never on a public deploy, nor where the Lab is switched off."""
    return not style.public() and os.environ.get("OPENDETECTOR_REDTEAM", "").lower() != "off"


def _undo(text: str) -> None:
    st.session_state[PENDING] = text
    st.session_state.pop(REVISED, None)


def _clear_picks() -> None:
    st.session_state[PICK_ROUND] = st.session_state.get(PICK_ROUND, 0) + 1


def _revised_note(result: dict) -> None:
    """What the last revise did, over the result of the text it made, with Undo."""
    done = st.session_state.get(REVISED)
    if not done or done["digest"] != result["digest"]:
        return
    if not done["changed"]:
        st.info(f"No rewrite of the {done['passages']} passage{'s' * (done['passages'] != 1)} read as more human "
                f"where it stands, so the text is as it was ({done['requests']} requests).", icon=":material/info:")
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        st.success(f"Revised {done['passages']} passage{'s' * (done['passages'] != 1)}: the whole text went from "
                   f"{done['before']:.0%} to {done['after']:.0%} AI likelihood in {done['requests']} requests"
                   + (f"; {done['still_ai']} still flagged where they stand" if done["still_ai"] else "")
                   + ". The revised text is in Source." + (" Saved as AI-labelled evaluation data."
                                                           if done["saved"] else ""),
                   icon=":material/auto_fix_high:", width="stretch")
        st.button("Undo", icon=":material/undo:", on_click=_undo, args=(done["undo"],), key="detect:undo")


def _revise_bar(result: dict, picked: list[int]) -> None:
    """Revise the picked sentences, or every Likely AI or flagged one, in place: the result is analyzed again here,
    so there's no going back and forth with the Lab. Writes with the Lab's style profile and models."""
    from . import redteam

    report, bands = result["report"], result["bands"]
    likely_ai = [i for i, band in enumerate(bands) if band == "likely_ai"]
    flagged = [i for i, band in enumerate(bands) if band in ("likely_ai", "needs_review")]
    choices = {"picked": picked, "likely_ai": likely_ai, "flagged": flagged}
    clicked = None
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if st.button(f"Revise selected ({len(picked)})", type="primary", icon=":material/auto_fix_high:",
                     disabled=not picked, key="detect:revise:picked",
                     help="Click sentences in the text to pick them, shift-click to pick a run."):
            clicked = "picked"
        if st.button(f"Revise all Likely AI ({len(likely_ai)})", icon=":material/smart_toy:", disabled=not likely_ai,
                     key="detect:revise:likely_ai", help="Every sentence highlighted as Likely AI."):
            clicked = "likely_ai"
        if len(flagged) > len(likely_ai) and st.button(
                f"Revise all flagged ({len(flagged)})", icon=":material/flag:", key="detect:revise:flagged",
                help="Every sentence highlighted as Likely AI or Needs review."):
            clicked = "flagged"
        if picked:
            st.button("Clear selection", type="tertiary", icon=":material/close:", on_click=_clear_picks,
                      key="detect:revise:clear")
    with st.expander("Revise settings · style profile and model", icon=":material/tune:"):
        st.caption("Each picked passage gets several rewrites a round in the style profile's voice, and the one "
                   "reading most human where it stands is kept if it beats the original and keeps its names and "
                   f"numbers, for up to {redteam.REVISE_ROUNDS} rounds. These are the Adversarial Lab's settings.")
        plan, conn = redteam.revise_settings(result["profile"])
    if clicked is None:
        return
    if problem := redteam.key_problem(conn):
        st.error(problem, icon=":material/key_off:")
        return
    status = st.status("Scoring the passages where they stand…", expanded=False)
    started = time.monotonic()

    def progress(now: dict) -> None:
        stage = now.get("stage")
        if stage == "asking":
            step = (f"waiting for {now['tries']} rewrites of each of {now['passages']} passage"
                    f"{'s' * (now['passages'] != 1)} from the model")
        elif stage == "whole":
            step = "scoring the whole text"
        else:
            step = f"the detector has read the rewrites of {now['tested']} of {now['passages']} passages"
        status.update(label=f"Round {now['round']} of up to {now['rounds']}: {step} · whole text at "
                            f"{now['likelihood']:.0%} · {redteam.duration(time.monotonic() - started)} so far")

    try:
        done = redteam.revise_chosen(plan, conn, report.text, report.sentences, choices[clicked], clicked, progress)
    except (redteam.APIError, OSError, ValueError) as exc:
        status.update(label="Revising stopped", state="error")
        st.error(str(exc), icon=":material/error:")
        return
    status.update(label="Revised", state="complete")
    st.session_state[REVISED] = {**done, "undo": report.text, "digest": _digest(done["text"])}
    st.session_state[PENDING] = done["text"]
    _clear_picks()
    st.rerun()


def show_result(result: dict, digest: str | None = None, revise: bool = False) -> None:
    """The verdict, the engines' readings and the tabs. With `digest`, says when the input has moved on. With
    `revise`, sentences can be picked in the highlights and revised in place (_revise_bar)."""
    report = result["report"]
    offered = _revising_offered()
    revise = revise and offered and bool(report.sentences) and report.ai_probability is not None
    if revise:
        _revised_note(result)
    elif offered and report.sentences and st.button("Revise flagged passages in the Lab", icon=":material/rule:"):
        st.session_state["keep:redteam:revise_text"] = report.text
        st.switch_page(nav.redteam_page())
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="od-section-title"><b>Result</b><span>{html.escape(result["source"])} · '
                f'{models.PROFILE_LABELS[result["profile"]]} build</span></div>')
    if digest is not None and result["digest"] != digest:
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
        foot = (f"{report.words:,} words · " + ("calibrated" if report.calibrated else "uncalibrated")
                + f" · cut-offs at {report.thresholds['review']:.0%} and {report.thresholds['ai']:.0%}")
        verdict_card(report.verdict, report.ai_probability, report.thresholds, report.calibrated, foot)
        _engine_readings(report, result["bands"])

    names = (["Highlights"] if report.ai_probability is not None else []) \
        + (["Sentences"] if report.sentences else []) + (["Sections"] if result["sections"] else []) \
        + (["File check"] if result.get("forensics") else []) + ["Details"]
    for tab, name in zip(st.tabs(names), names, strict=True):
        with tab:
            if name == "Highlights":
                if report.sentences:
                    mode = st.segmented_control("Highlight", HIGHLIGHTS, key=keep("keep:detect:highlight",
                                                                                  HIGHLIGHTS[0]),
                                                required=True, label_visibility="collapsed")
                    page = highlighted_text(report.text, report.sentences, result["bands"], report.thresholds,
                                            flagged_only=mode == HIGHLIGHTS[1], pickable=revise)
                    if revise:
                        bar = st.container()
                        picked = pickable_highlights(
                            page, f"detect:pick:{result['digest'][:16]}:{st.session_state.get(PICK_ROUND, 0)}")
                        with bar:
                            _revise_bar(result, picked)
                    else:
                        st.html(page, unsafe_allow_javascript=True)
                    st.caption("Stronger shades sit deeper in their band. Likely AI is underlined, Needs review "
                               "dotted. Each sentence takes the average score of the ~100-word windows around "
                               "it, so sentences next to a strongly AI-like passage get pulled toward it. "
                               "Highlights show where to look; the document score is the number to go by.")
                else:
                    st.caption("Sentence highlights are off. Switch them on under Engines and analyze again.")
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
                        format="percent", min_value=0, max_value=1, width="small", color=style.tokens()["accent"]),
                        "Band": st.column_config.TextColumn(width="small")},
                )
            elif name == "Sections":
                _sections_table(result["sections"], result["min_words"], report.thresholds)
            elif name == "File check":
                _file_check(result["forensics"])
            else:
                _details(result)



def render() -> None:
    if (pending := st.session_state.pop(PENDING, None)) is not None:  # before Source draws its text box
        st.session_state["keep:detect:source"] = SOURCES[0]
        st.session_state["keep:detect:text"] = pending
        st.session_state[AUTORUN] = True
    profile = models.active_profile()
    names = [models.ENGINES[name][0] for name in models.engines_of(profile)]
    on = sum(st.session_state.get(f"keep:detect:engine:{name}", True) for name in models.engines_of(profile)) \
        if len(names) > 1 else len(names)
    style.page_header("Check a text", "Paste text or upload a document, then press Analyze.",
                      f"{models.PROFILE_LABELS[profile]} build · {on} of {len(names)} engines on", ready=on > 0)

    setup, output = st.columns([1, 1.45], gap="large")
    with setup:
        with style.group("Source", "01"):
            text, source, sections = _input()
            words = count_words(text) if text.strip() else 0
            if words:
                st.caption(f"{words:,} words" + ("" if words >= MIN_WORDS else
                                                 f" · a verdict needs at least {MIN_WORDS}"))
        enabled, sentences = _engines(profile)
        clicked = st.button("Analyze", type="primary", icon=":material/search:", width="stretch")
        st.caption(":material/balance: A detector score is statistical evidence, not proof of authorship. Don't "
                   "use it as the only basis for an academic integrity decision.")

    with output:
        if clicked or st.session_state.pop(AUTORUN, False):
            if not text.strip():
                st.warning("Paste some text or upload a file first.")
            elif not enabled:
                st.warning("Switch on at least one engine under Engines.")
            else:
                _analyze(text, source, sections, profile, enabled, sentences)

        if result := st.session_state.get(RESULT):
            show_result(result, _digest(text), revise=True)
        else:
            st.html('<div class="od-empty"><b>No result yet</b>Paste at least 150 words of prose or upload a '
                    "document on the left, then press Analyze. The verdict, each engine's reading and the "
                    "sentences that drive the score appear here.</div>")
