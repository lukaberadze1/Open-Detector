"""The Red team page: generate labelled AI samples and see how the detector scores them.

Everything it writes is AI-labelled evaluation data. Each sample is appended to
data/essays/local_redteam.jsonl with label 1 and exported under data/essays/exports/. Batches, which
regenerate every answer box or text block of an uploaded document, run on a worker thread and save
each block as they go (eval/redteam_batch.py), so reruns, page switches and closed tabs don't end them.

On a public deploy (OPENDETECTOR_PUBLIC=on) each visitor brings their own Gemini key and style
profile, nothing reads the host's keys, and each visitor's samples, profiles and batches are kept in
a folder of their own under data/visitors/ for as long as their browser session lasts.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import streamlit as st

from eval.generate_essays import (
    DEFAULT_MODEL,
    GEMINI_API_URL,
    LOCAL_MODEL,
    LOCAL_URL,
    THINKING_LEVELS,
    APIError,
    answer_token_limit,
    clean_essay,
    generate_with_fallback,
    is_local,
    load_style_profile,
    local_context_size,
    local_model_problem,
    style_instructions,
    unload_local_model,
)
from eval import generation_settings
from eval.generation_settings import MAX_POINTS, MEASURES
from eval.redteam_batch import (
    GROUP_SCHEMA,
    BatchWorker,
    Checkpoint,
    StopRequested,
    batch_key,
    block_prompt,
    group_prompt,
    parse_group_reply,
    person_instruction,
    target_words,
)
from eval.evasion_mode import PROFILE_ID, PROFILE_INSTRUCTIONS, generate_until_target
from opendetector import DetectorConfig
from opendetector.config import DATA_ROOT
from opendetector.documents import (
    AnswerSlot,
    DocumentSection,
    extract_answer_slots,
    extract_document,
    replace_document_section,
    rewritable_blocks,
)
from opendetector.env import load_env
from opendetector.segment import count_words

from . import detect, models, nav, style
from .components import VERDICTS, Upload, keep, persistent_upload, priorities_chart, verdict_badge

ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = DATA_ROOT / "data/styles"
DEFAULT_STYLE = "alexandria_profile"  # picked first when it's there
OUTPUT_PATH = DATA_ROOT / "data/essays/local_redteam.jsonl"
EXPORT_DIR = DATA_ROOT / "data/essays/exports"
BATCH_DIR = DATA_ROOT / "data/essays/batches"
VISITOR_DIR = DATA_ROOT / "data/visitors"  # a public deploy's per-visitor folders
MAX_STYLE_BYTES = 500_000
EVASION_TARGET = 0.05  # the AI likelihood robustness mode aims below
DEFAULT_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)
DOCUMENT_TYPES = ("md", "docx", "pptx")
MIME_TYPES = {
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
TASKS = {
    "sample": ("New sample", "Write a new sample from your instructions, with any uploaded document as source "
                             "material."),
    "section": ("Rewrite a section", "Write a replacement for one section of the uploaded document, with the "
                                     "rest of it as context."),
    "answers": ("Fill answer boxes", "Write a new answer in every right-hand answer box of the document's "
                                     "question/answer tables or slides, then export a filled copy."),
    "blocks": ("Rewrite every block", "Rewrite each text block in turn, keeping headings, titles, questions and "
                                      "tables, then export the rewritten file."),
}
RESULT = "last_generated_result"
POINTS = "redteam:points"  # the priorities' points, kept for the browser session
PRIORITIES = {
    "detector": (":material/visibility_off:", "Gets past the detector"),
    "style": (":material/draw:", "Sounds like you"),
    "speed": (":material/bolt:", "Speed"),
    "quota": (":material/savings:", "Saves quota"),
}
NOTICE = "redteam:notice"  # how the last batch ended, shown until the next run starts
# The largest context window a local model gets on a machine with 8 GB or less. gemma3:4b ran with
# 4,096 tokens and crashed Ollama with 32,768 on an 8 GB Mac; a batch whose prompts need more than this
# leaves the local model out and stops when the hosted models are spent.
SMALL_MACHINE_LOCAL_CONTEXT = 8192
# When every model a batch has left is busy or rate-limited, it waits and asks again: 30 s, then 60 s
# and so on, five times (7.5 minutes in all) before it stops.
BUSY_RETRIES, BUSY_WAIT = 5, 30


# Checked on every rerun to show in the sidebar, so the answer is kept for a few seconds.
local_fallback_problem = st.cache_data(ttl=15, show_spinner=False)(local_model_problem)


@st.cache_resource
def running_batches() -> dict:
    """Batch workers by checkpoint file, shared by every session and kept across reruns."""
    return {}


def public() -> bool:
    """Whether this is a public deploy, where visitors bring their own key and see only their own work:
    OPENDETECTOR_PUBLIC when it is set, otherwise whether this runs on Streamlit Community Cloud, which
    keeps apps under /mount/src. That way an app started from app.py there never serves the host's keys."""
    value = os.environ.get("OPENDETECTOR_PUBLIC", "").strip().lower()
    if value:
        return value in ("1", "on", "true", "yes")
    return ROOT.parts[:3] == ("/", "mount", "src")


def packaged() -> bool:
    """Whether this is the downloadable Mac app (deploy/macos/build_app.py --standalone). It keeps its
    user's key and samples like a local run, but serves people this project doesn't know, so it offers
    no local text model by default."""
    return os.environ.get("OPENDETECTOR_PACKAGED", "").strip().lower() in ("1", "on", "true", "yes")


def _shown(path: Path) -> str:
    """A saved file's path as the page shows it: inside the project folder on a local run, or in full,
    starting from ~, when the app keeps its data elsewhere."""
    if path.is_relative_to(DATA_ROOT) and DATA_ROOT == ROOT:
        return str(path.relative_to(DATA_ROOT))
    return f"~/{path.relative_to(Path.home())}" if path.is_relative_to(Path.home()) else str(path)


@dataclass(frozen=True)
class Store:
    """Where the page keeps samples, exports, batch checkpoints and style profiles."""

    output: Path
    exports: Path
    batches: Path
    styles: Path


def _store() -> Store:
    """The project's data folders, or on a public deploy a folder of this browser session's own."""
    if not public():
        return Store(OUTPUT_PATH, EXPORT_DIR, BATCH_DIR, STYLE_DIR)
    root = VISITOR_DIR / st.session_state.setdefault("redteam:visitor", uuid.uuid4().hex)
    return Store(root / "local_redteam.jsonl", root / "exports", root / "batches", root / "styles")


def _own_batches() -> dict[str, BatchWorker]:
    """The batches this session may see: those whose checkpoints are in its own store."""
    batches = _store().batches
    return {path: worker for path, worker in running_batches().items() if Path(path).parent == batches}


@dataclass
class Connection:
    base_url: str
    model: str
    fallback_models: list[str]
    local_model: str
    local_url: str
    api_key_env: str
    api_key: str


@dataclass
class Plan:
    """What the page is set to generate."""

    prompt: str
    upload: Upload | None
    sections: list[DocumentSection]
    answer_slots: list[AnswerSlot]
    blocks: list[DocumentSection]
    task: str
    profile: str
    fill_answer_boxes: bool = False
    rewrite_all_blocks: bool = False
    target_section: DocumentSection | None = None
    context_sections: list[DocumentSection] = field(default_factory=list)
    export_document: bool = False
    words: int = 350
    answer_words: int = 100
    match_existing_lengths: bool = True
    temperature: float = 0.8
    thinking: str = "low"
    group_size: int = 6  # blocks per request in a batch
    parallel: int = 2  # a batch's requests at once
    model: str | None = None  # the model to start with, when the priorities picked one
    style_path: Path | None = None
    score: bool = True
    evasion: bool = False

    @property
    def batch(self) -> bool:
        return self.fill_answer_boxes or self.rewrite_all_blocks

    @property
    def use_profile(self) -> bool:
        return self.style_path is not None

    @property
    def style_name(self) -> str:
        return self.style_path.stem if self.style_path else "plain"

    @property
    def markdown(self) -> bool:
        return self.upload is not None and self.upload.suffix == ".md"

    @property
    def mode_tag(self) -> str:
        if self.rewrite_all_blocks:
            return "full_markdown" if self.markdown else "full_document"
        return "filled_answers"

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.upload.data).hexdigest()[:16]

    @property
    def context_text(self) -> str:
        return "\n\n".join(f"[{section.label}]\n{section.text}" for section in self.sections)


def duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600:.0f} h {seconds % 3600 / 60:.0f} min"
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _keep_awake() -> subprocess.Popen | None:
    """Keep the Mac from going to sleep while a batch runs: until the returned process is ended, or
    this server stops. Nothing happens on other systems. The Mac app (deploy/macos/launcher.py)
    looks for this caffeinate child to ask before quitting during a batch."""
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    except OSError:
        return None


def _notify(message: str) -> None:
    """A macOS notification, so a long batch can be left to run. Nothing happens on other systems."""
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", "on run argv", "-e",
                        'display notification (item 1 of argv) with title "Open Detector" sound name "Glass"',
                        "-e", "end run", message], capture_output=True, timeout=10, check=False)


def score_text(text: str, profile: str) -> dict:
    """A block's score record. Text under the detector's minimum is not run through it."""
    if count_words(text) < DetectorConfig.for_profile(profile).min_words:
        result = {"ai_probability": None, "verdict": "insufficient_text", "calibrated": None}
    else:
        analysis = models.get_detector(profile).analyze(text, sentences=False)
        result = {
            "ai_probability": analysis.ai_probability
            if analysis.verdict not in ("insufficient_text", "not_prose")
            else None,
            "verdict": analysis.verdict,
            "calibrated": analysis.calibrated,
        }
    return {**result, "detector_profile": profile}


# -- settings -------------------------------------------------------------------------


def _connection() -> Connection:
    """The sidebar: which text models to use, and whether they can run."""
    with st.sidebar:
        st.subheader("Text model")
        if public():
            return _public_connection()
        status = st.container()
        with st.expander("Connection", icon=":material/settings:"):
            base_url = st.text_input("API URL", key=keep("keep:redteam:base_url", GEMINI_API_URL),
                                     help="Gemini uses its Interactions API. Any OpenAI-compatible endpoint "
                                          "works too.")
            model = st.text_input("Model", key=keep("keep:redteam:model", DEFAULT_MODEL))
            fallback_text = st.text_area(
                "Fallback models, in order", key=keep("keep:redteam:fallbacks", "\n".join(DEFAULT_FALLBACK_MODELS)),
                height=130,
                help="Gemini uses these current text models after quota or temporary errors. Models unavailable "
                     "to your account are skipped. Edit or clear this list for another OpenAI-compatible provider.",
            )
            local_model = st.text_input(
                "Local fallback model (Ollama)",
                key=keep("keep:redteam:local_model", "" if packaged() else LOCAL_MODEL),
                help="Used after every model above fails. Runs on this computer with no key or quota; needs "
                     "`ollama serve` running and the model fetched with `ollama pull`. Clear to turn it off.",
            ).strip()
            local_url = st.text_input("Local fallback URL", key=keep("keep:redteam:local_url", LOCAL_URL)).strip()
            api_key_env = st.text_input("API key environment variable",
                                        key=keep("keep:redteam:key_env", "GEMINI_API_KEY"))
        fallback_models = [line.strip() for line in fallback_text.replace(",", "\n").splitlines() if line.strip()]
        api_key = os.environ.get(api_key_env, "")
        with status:
            st.caption(f":material/smart_toy: **{model}**, then {len(fallback_models)} fallback models"
                       + (f" and {local_model} locally." if local_model else "."))
            if api_key:
                st.caption(f":material/key: Using your key in {api_key_env}.")
            else:
                st.caption(f":material/key_off: No {api_key_env} yet. Local endpoints work without a key.")
                st.page_link(nav.guide_page(), label="Add your Gemini API key", icon=":material/key:")
            if local_model:
                if problem := local_fallback_problem(local_url, local_model):
                    st.caption(f":orange[:material/cloud_off: The local fallback can't run: {problem}]")
                else:
                    st.caption(f":material/check_circle: Local fallback ready: {local_model}.")
    return Connection(base_url, model, fallback_models, local_model, local_url, api_key_env, api_key)


def _public_connection() -> Connection:
    """The sidebar on a public deploy: Gemini only, with the visitor's own key. The key stays in this
    browser session's memory; it is never saved, and the host's own keys are never used."""
    api_key = st.text_input(
        "Your Gemini API key", type="password", key=keep("keep:redteam:api_key", ""),
        help="Free at aistudio.google.com/apikey. It is used only for your requests, held only while this "
             "browser session lasts and never saved.",
    ).strip()
    with st.expander("Models", icon=":material/settings:"):
        model = st.text_input("Model", key=keep("keep:redteam:model", DEFAULT_MODEL))
        fallback_text = st.text_area(
            "Fallback models, in order", key=keep("keep:redteam:fallbacks", "\n".join(DEFAULT_FALLBACK_MODELS)),
            height=130, help="Used in turn after quota or temporary errors. Models unavailable to your key are "
                             "skipped.",
        )
    fallback_models = [line.strip() for line in fallback_text.replace(",", "\n").splitlines() if line.strip()]
    if api_key:
        st.caption(f":material/key: Writing with **{model}** and your key, then {len(fallback_models)} fallback "
                   "models.")
    else:
        st.caption(":material/key_off: Paste your Gemini API key to generate samples. Get one free at "
                   "[aistudio.google.com/apikey](https://aistudio.google.com/apikey).")
    return Connection(GEMINI_API_URL, model, fallback_models, "", "", "GEMINI_API_KEY", api_key)


def _setup(profile: str, conn: Connection) -> Plan:
    """The left-hand column: source, task, length, style and scoring."""
    st.markdown("##### :material/edit_note: Source")
    prompt = st.text_area("Instructions", key=keep("keep:redteam:prompt", ""), height=130,
                          placeholder="Describe what the sample should cover: the assignment, the audience, "
                                      "the angle.")
    upload = persistent_upload(
        "Source document (optional)", "redteam:file", DOCUMENT_TYPES,
        help="A Markdown, Word or PowerPoint file. Its text is context for the sample, and it lets you rewrite "
             "its sections or fill its answer boxes. The file itself is never changed.",
    )
    sections, slots, blocks = [], [], []
    if upload:
        try:
            sections = extract_document(upload.name, upload.data)
            slots = extract_answer_slots(upload.name, upload.data)
            blocks = rewritable_blocks(upload.name, upload.data)
        except (ImportError, ValueError) as exc:
            st.error(f"Could not read {upload.name}: {exc}")
            upload = None
        else:
            with st.expander(f"Extracted text · {len(sections)} sections", icon=":material/article:"):
                for section in sections:
                    st.markdown(f"**{section.label}**")
                    st.text(section.text)
    plan = Plan(prompt, upload, sections, slots, blocks, "sample", profile)

    st.markdown("##### :material/task_alt: Task")
    unavailable = _unavailable_tasks(upload, sections, slots, blocks)
    options = [task for task in TASKS if task not in unavailable]
    task_key = keep("keep:redteam:task", "sample")
    if st.session_state[task_key] not in options:
        st.session_state[task_key] = "sample"
    plan.task = st.segmented_control("Task", options, format_func=lambda task: TASKS[task][0], key=task_key,
                                     required=True, label_visibility="collapsed")
    st.caption(TASKS[plan.task][1])
    if not upload:
        st.caption(":material/upload_file: Upload a Markdown, Word or PowerPoint document to rewrite a section, "
                   "fill its answer boxes or rewrite every block.")
    else:
        for task, reason in unavailable.items():
            st.caption(f":material/block: **{TASKS[task][0]}** isn't available for this file: {reason}")

    plan.context_sections = sections
    if plan.task == "section":
        labels = [section.label for section in sections]
        section_key = keep("keep:redteam:section", labels[0])
        if st.session_state[section_key] not in labels:
            st.session_state[section_key] = labels[0]
        target_label = st.selectbox("Section to rewrite", labels, key=section_key)
        plan.target_section = next(section for section in sections if section.label == target_label)
        others = [section for section in sections if section.label != target_label]
        context_labels = st.multiselect(
            "Other sections to give as context", [section.label for section in others],
            default=[section.label for section in others], key=f"redteam:context:{target_label}",
            help="Questions and earlier answers help the model keep the replacement consistent.",
        )
        plan.context_sections = [section for section in others if section.label in context_labels]
        st.caption("The selected section is treated as the previous AI answer. The uploaded file stays unchanged.")
        plan.export_document = st.toggle("Also export a copy of the document with this section replaced",
                                         key=keep("keep:redteam:export", False),
                                         help="Exports a new AI-labelled Markdown, DOCX or PPTX copy.")
    elif plan.task == "answers":
        plan.fill_answer_boxes = True
        with st.expander(f"Detected answer boxes · {len(slots)}", icon=":material/table_rows:"):
            for slot in slots:
                st.markdown(f"**{slot.label}**")
                st.write(f"Question: {slot.question}")
                st.write(f"Current answer: {slot.previous_answer or '(blank)'}")
    elif plan.task == "blocks":
        plan.rewrite_all_blocks = plan.export_document = True
        if slots:
            plan.fill_answer_boxes = st.toggle(f"Also fill the {len(slots)} answer boxes",
                                               key=keep("keep:redteam:also_answers", False))
        st.caption(f"Rewrites {len(blocks)} text blocks. Headings, titles, questions and tables stay as they are"
                   + (" unless you fill the answer boxes too." if slots else "."))
        with st.expander(f"Blocks to rewrite · {len(blocks)}", icon=":material/segment:"):
            for block in blocks:
                st.markdown(f"**{block.label}**")
                st.text(block.text if len(block.text) <= 300 else block.text[:300] + "…")

    st.markdown("##### :material/tune: Length, style and scoring")
    if plan.batch:
        plan.match_existing_lengths = st.toggle("Match the length of the existing text",
                                                key=keep("keep:redteam:match", True))
        plan.answer_words = st.slider(
            "Words per block", min_value=20, max_value=500, step=10, key=keep("keep:redteam:answer_words", 100),
            help="For blank blocks, or for every block when not matching the existing lengths.",
        )
    else:
        plan.words = st.slider("Approximate length", min_value=150, max_value=1500, step=50,
                               key=keep("keep:redteam:words", 350), format="%d words")
    plan.temperature = st.slider("Creativity", 0.0, 1.5, step=0.1, key=keep("keep:redteam:temperature", 0.8),
                                 help="The model's temperature.")
    styles = {path.stem: path for path in sorted(_store().styles.glob("*.json"))}
    use_style = st.toggle(
        "Write in a style profile", key=keep("keep:redteam:style", True), disabled=not styles,
        help="Sends a style profile as the system prompt. A style profile imitates writing patterns; it can't "
             "reproduce a writer exactly.",
    )
    if styles:
        style_key = keep("keep:redteam:style_name", DEFAULT_STYLE)
        if st.session_state[style_key] not in styles:
            st.session_state[style_key] = DEFAULT_STYLE if DEFAULT_STYLE in styles else next(iter(styles))
        name = st.selectbox("Style profile", list(styles), key=style_key, disabled=not use_style,
                            help="The profiles you added in this browser session." if public() else
                                 "Every JSON file in data/styles/. Save another profile there and it shows up here.")
        plan.style_path = styles[name] if use_style else None
    else:
        st.caption("No style profile yet, so samples use the model's own style. Add one below to write in it.")
    st.file_uploader(
        "Add a style profile", type="json", key="redteam:style_upload", on_change=_add_style_profile,
        help="A JSON style profile. " + ("It is kept only for this browser session." if public() else
                                         "It is saved in data/styles/."),
    )
    if style_notice := st.session_state.pop("redteam:style_notice", None):
        getattr(st, style_notice[0])(style_notice[1])
    plan.score = st.toggle("Score with the detector", key=keep("keep:redteam:score", True),
                           help=f"Uses the {models.PROFILE_LABELS[profile]} build chosen in the sidebar. The "
                                "first score loads its models.")
    if not public():  # the packaged app allows labelled robustness samples, but public deploys do not
        evasion = st.toggle(
            "Detector robustness mode · aim below 5%", key=keep("keep:redteam:evasion", False),
            disabled=False,
            help="For labelled red-team samples, try up to three AI drafts and keep the lowest detector score. "
                 "Available for single samples and document batches; it does not guarantee a score under 5%.",
        )
        plan.evasion = evasion
        if plan.evasion:
            plan.score = True
    if plan.evasion:
        st.caption("Every draft is saved as AI-labelled evaluation data; the kept one records the mode and score."
                   + (" In a batch, each block is drafted and scored on its own, one at a time." if plan.batch
                      else ""))
    _priorities(plan, conn)
    return plan


def _spend(measure: str, change: int) -> None:
    st.session_state[POINTS][measure] += change


def _priorities(plan: Plan, conn: Connection) -> None:
    """What the run favours. Points spread over the detector, the style profile, speed and quota pick
    a setting from the measured table (eval/generation_settings.py); without one, or by choice, the
    thinking level, blocks per request and requests at once are set by hand."""
    st.markdown("##### :material/joystick: Priorities")
    table = generation_settings.load_table()
    models_on_offer = [name for name in (conn.model, *conn.fallback_models) if name]
    if table is None:
        st.caption("Set by hand until the settings are measured: run `python -m eval.tune_generation` on a document "
                   "to let points pick them.")
    elif not st.toggle("Set by hand", key=keep("keep:redteam:by_hand", False),
                       help="Choose the thinking level, blocks per request and requests at once yourself."):
        points = st.session_state.setdefault(POINTS, dict(generation_settings.EVEN))
        left = generation_settings.POINTS - sum(points.values())
        for measure in MEASURES:
            icon, label = PRIORITIES[measure]
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"{icon} {label}", width="stretch")
                st.markdown(f":violet[{'●' * points[measure]}]:gray[{'○' * (MAX_POINTS - points[measure])}]",
                            width="content")
                st.button("\u2212", key=f"redteam:less:{measure}", type="tertiary",  # a minus sign
                          on_click=_spend, args=(measure, -1), disabled=points[measure] == 0,
                          help=f"One point less on {label.lower()}")
                st.button("+", key=f"redteam:more:{measure}", type="tertiary",
                          on_click=_spend, args=(measure, 1), disabled=not left or points[measure] == MAX_POINTS,
                          help=f"One point more on {label.lower()}")
        st.caption(f"{left} of {generation_settings.POINTS} points left"
                   + (": take one from a priority to give it to another." if not left else ".")
                   + " Even points are the measured balance.")
        blocks = len(_batch_jobs(plan)) if plan.batch else 1
        if setting := generation_settings.choose(table, points, models_on_offer, plan.batch, blocks):
            plan.model, plan.thinking = setting.model, setting.thinking
            plan.group_size, plan.parallel = setting.group_size, setting.parallel
            chart, readout = st.columns([1, 1.6], vertical_alignment="center")
            chart.html(priorities_chart(points, MAX_POINTS))
            with readout:
                _readout(plan, table, setting, blocks)
            return
        st.caption("None of the measured settings uses the models in the sidebar, so set them by hand.")
    plan.thinking = st.select_slider(
        "Model thinking", THINKING_LEVELS, key=keep("keep:redteam:thinking", Plan.thinking),
        format_func=str.capitalize,
        help="How long Gemini reasons before it writes. Less is faster and spends less quota, and short rewrites "
             "rarely need more than low. A model without the level gets the next one up; other providers "
             "use their own.",
    )
    if plan.batch:
        with st.container(horizontal=True):
            plan.group_size = st.slider(
                "Blocks per request", 1, 10, key=keep("keep:redteam:group_size", Plan.group_size),
                help="Neighbouring blocks rewritten together in one request, so the style profile and the "
                     "document are sent once per group rather than once per block. Any block a reply leaves out "
                     "is written again on its own.",
            )
            plan.parallel = st.slider(
                "Requests at once", 1, 5, key=keep("keep:redteam:parallel", Plan.parallel),
                help="How many groups are written at the same time. More finish sooner, until Gemini's "
                     "per-minute limit sends some to the fallback models. A local model takes one at a time.",
            )


def _readout(plan: Plan, table: dict, setting: generation_settings.Setting, blocks: int) -> None:
    """The picked setting and what it should cost this document, from the measured table."""
    context = len(plan.context_text)
    if plan.use_profile:
        with contextlib.suppress(OSError, ValueError):
            context += len(style_instructions(load_style_profile(plan.style_path)))
    estimate = generation_settings.estimate(setting, blocks, table, context)
    st.markdown(f":material/auto_awesome: **{setting.model}** · {setting.thinking} thinking"
                + (f" · {setting.group_size} blocks per request · {setting.parallel} at once" if plan.batch else ""))
    scores = [f"AI likelihood {setting.ai:.0%}" if setting.ai is not None else None,
              f"style match {setting.style:.0f}%" if setting.style is not None else None]
    requests = max(1, round(estimate["requests"]))
    limit = estimate["daily_limit"]
    allowance = (f" ({min(1.0, requests / limit):.0%} of its {limit:,} free requests a day)" if limit else
                 " (its daily limit isn't known yet)")
    st.caption(f"About {duration(estimate['seconds'])}, {requests} request{'s' * (requests != 1)}{allowance} and "
               f"{estimate['tokens'] / 1000:,.0f}k tokens for this {'document' if plan.batch else 'sample'}. "
               + (f"{'Measured' if setting.measured else 'Predicted'}: {', '.join(filter(None, scores))}."
                  if any(scores) else ""))


def _add_style_profile() -> None:
    """Save an uploaded style profile to the store and pick it. Runs before the page redraws, so it
    can set the picker's value."""
    upload = st.session_state.get("redteam:style_upload")
    if upload is None:
        return
    data = upload.getvalue()
    name = re.sub(r"[^\w-]+", "_", Path(upload.name).stem).strip("_") or "style_profile"
    try:
        if len(data) > MAX_STYLE_BYTES:
            raise ValueError(f"it is over {MAX_STYLE_BYTES // 1000} KB")
        if not isinstance(json.loads(data), dict):
            raise ValueError("it must hold a JSON object")
    except ValueError as exc:  # also bad JSON and bad text encodings
        st.session_state["redteam:style_notice"] = ("error", f"{upload.name} isn't a style profile: {exc}.")
        return
    path = _store().styles / f"{name}.json"
    if path.is_file() and path.read_bytes() != data:
        st.session_state["redteam:style_notice"] = (
            "error", f"There is already a different profile named {name}. Rename the file and add it again.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    st.session_state["keep:redteam:style_name"] = name
    st.session_state["keep:redteam:style"] = True
    st.session_state["redteam:style_notice"] = ("success", f"Added {name}, and picked it.")


def _unavailable_tasks(upload: Upload | None, sections: list, slots: list, blocks: list) -> dict[str, str]:
    """The document tasks the uploaded file doesn't support, with the reason for each. Without a file,
    none of them is available."""
    if not upload:
        return {"section": "", "answers": "", "blocks": ""}
    reasons = {}
    if not sections:
        reasons["section"] = f"found no text in {upload.name}."
    if not slots:
        reasons["answers"] = (
            "found no question/answer pairs. It looks for tables with the question in the left column and the "
            "answer in the right one"
            + (", or a text box on the left half of a slide with one level with it on the right half."
               if upload.suffix == ".pptx" else ".")
        )
    if not blocks:
        reasons["blocks"] = "found no prose to rewrite outside headings, titles, questions and tables."
    return reasons


def _problem(plan: Plan, conn: Connection) -> str | None:
    """Why the plan can't run, if it can't."""
    if not plan.prompt.strip() and not plan.sections:
        return "Write some instructions or upload a Markdown, Word or PowerPoint document first."
    if plan.task == "section" and plan.target_section is None:
        return "Choose a document section to rewrite."
    if not conn.api_key and public():
        return "Paste your Gemini API key in the sidebar first."
    if not conn.api_key and not conn.base_url.strip().startswith(("http://localhost", "http://127.0.0.1")):
        return "Add your Gemini API key on the Guide page, or use a local model endpoint."
    return None


# -- batches --------------------------------------------------------------------------


def _batch_jobs(plan: Plan) -> list[dict]:
    jobs = []
    if plan.fill_answer_boxes:
        jobs.extend(
            {
                "kind": "answer_box",
                "label": slot.label,
                "question": slot.question,
                "previous": slot.previous_answer,
                "target": slot.target,
            }
            for slot in plan.answer_slots
        )
    if plan.rewrite_all_blocks:
        jobs.extend(
            {
                "kind": "markdown_block" if plan.markdown else "text_block",
                "label": section.label,
                "question": "",
                "previous": section.text,
                "target": section,
            }
            for section in plan.blocks
        )
    # Labels can repeat (two shapes with the same name), so a block's position is part of its key.
    for index, job in enumerate(jobs, 1):
        job["key"] = f"{index:03d} {job['kind']} {job['label']}"
    return jobs


def _checkpoint_path(plan: Plan) -> Path:
    # Batches saved before a profile could be chosen all used the default one, so only another
    # profile adds its name to the key, and those batches still resume.
    style = [plan.style_name] if plan.use_profile and plan.style_name != DEFAULT_STYLE else []
    return _store().batches / (
        batch_key(plan.source_hash, plan.fill_answer_boxes, plan.rewrite_all_blocks, plan.prompt.strip(),
                  plan.use_profile, plan.answer_words, plan.match_existing_lengths, *style) + ".json"
    )


def _export_batch(plan: Plan, jobs: list[dict], checkpoint: Checkpoint, partial: bool) -> None:
    """Export every generated block: a copy of the document with those blocks replaced (in a partial
    export the rest stay as uploaded) and a text file of the answers. Blocks that are final, scored or
    with scoring off, are appended to the evaluation records, once each."""
    keys = [job["key"] for job in jobs]
    generated = set(checkpoint.generated(keys))
    items = [{**job, **checkpoint.blocks[job["key"]]} for job in jobs if job["key"] in generated]
    if not items:
        return
    upload, source_hash, mode_tag = plan.upload, plan.source_hash, plan.mode_tag
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{mode_tag}_partial" if partial else mode_tag
    store = _store()
    store.exports.mkdir(parents=True, exist_ok=True)
    document_bytes = document_name = document_path = None
    try:
        document_bytes = upload.data
        for item in sorted(items, key=lambda value: value["target"].locator[0], reverse=True):
            document_bytes = replace_document_section(upload.name, document_bytes, item["target"], item["text"])
        document_name = f"AI_REDTEAM_{tag}_{timestamp}_{source_hash}{upload.suffix}"
        (store.exports / document_name).write_bytes(document_bytes)
        document_path = _shown(store.exports / document_name)
    except (ImportError, ValueError, IndexError, KeyError) as exc:
        document_bytes = document_name = None
        st.warning(f"The answers are saved, but the document copy could not be exported: {exc}")

    attack = f"{mode_tag}+{plan.style_name}"
    rows = []
    for item in items:
        if item.get("recorded") or (plan.score and "score" not in item):
            continue
        question_material = f"Target: {item['label']}\n"
        if item["question"]:
            question_material += f"Question:\n{item['question']}\n"
        question_material += (
            f"Previous text:\n{item['previous']}\n\nAdditional instructions:\n"
            f"{plan.prompt.strip()}\n\nDocument context:\n{plan.context_text}"
        )
        record_hash = hashlib.sha256(item["key"].encode()).hexdigest()[:8]
        rows.append({
            "id": f"local-{checkpoint.meta['created']}-{source_hash}-{record_hash}-{item['model']}-{attack}",
            "text": item["text"],
            "label": 1,
            "dataset": "local_redteam",
            "domain": {"answer_box": "document_answer_box", "markdown_block": "markdown_text_block"}.get(
                item["kind"], "document_text_block"),
            "model": item["model"],
            "attack": attack,
            "group": "ai",
            "group_key": f"document-{source_hash}",
            "prompt": question_material,
            "source_file": upload.name,
            "target_section": item["label"],
            "detector_profile": None,
            **(item.get("score") or {}),
        })
        checkpoint.blocks[item["key"]]["recorded"] = True
    # The whole regenerated text is a sample of its own: unlike most single blocks, it is long enough
    # for the detector and for the evaluation, which drops texts under 150 words.
    document_score = None if partial else checkpoint.meta.get("document_score")
    if document_score and not checkpoint.meta.get("document_recorded"):
        models_used = {item["model"] for item in items}
        model_tag = models_used.pop() if len(models_used) == 1 else "mixed"
        rows.append({
            "id": f"local-{checkpoint.meta['created']}-{source_hash}-document-{model_tag}-{attack}",
            "text": "\n\n".join(item["text"] for item in items),
            "label": 1,
            "dataset": "local_redteam",
            "domain": "document_combined",
            "model": model_tag,
            "attack": attack,
            "group": "ai",
            "group_key": f"document-{source_hash}",
            "prompt": f"Every regenerated block of {upload.name}, in document order.",
            "source_file": upload.name,
            "target_section": None,
            **{key: value for key, value in document_score.items() if key != "words"},
        })
        checkpoint.meta["document_recorded"] = True
    if rows:
        store.output.parent.mkdir(parents=True, exist_ok=True)
        with store.output.open("a", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        checkpoint.save()

    answers_text = [
        f"{item['label']}\n{'Answer' if item['kind'] == 'answer_box' else 'Regenerated text'}: {item['text']}"
        for item in items
    ]
    text_name = f"AI_REDTEAM_{tag}_{timestamp}_{source_hash}.txt"
    text_attachment = (
        "Open Detector | AI-labelled red-team answers\n"
        f"Source document: {upload.name}\n"
        f"Regenerated blocks: {len(items)} of {len(jobs)}\n\n" + "\n\n".join(answers_text) + "\n"
    )
    (store.exports / text_name).write_text(text_attachment, encoding="utf-8")
    noun = ("Markdown text blocks" if plan.markdown else "text blocks") if plan.rewrite_all_blocks else "answer boxes"
    st.session_state[RESULT] = {
        "essay": "\n\n".join(answers_text),
        "detect_text": "\n\n".join(item["text"] for item in items),
        "row": rows[0] if rows else {"model": items[0]["model"], "attack": attack},
        "score": (
            {
                "probability": document_score["ai_probability"],
                "verdict": document_score["verdict"],
                "calibrated": document_score["calibrated"],
                "words": document_score["words"],
            }
            if document_score
            else None
        ),
        "score_label": "Whole regenerated document",
        "attachment": text_attachment,
        "filename": text_name,
        "path": _shown(store.exports / text_name),
        "document_attachment": document_bytes,
        "document_filename": document_name,
        "document_path": document_path,
        "batch_summary": (
            f"{'Regenerated' if plan.rewrite_all_blocks else 'Filled'} {len(items)} of {len(jobs)} {noun}"
            + (" so far" if partial else "")
        ),
        "batch_answers": [
            {
                "label": item["label"],
                "text": item["text"],
                "verdict": (item.get("score") or {}).get("verdict"),
                "probability": (item.get("score") or {}).get("ai_probability"),
            }
            for item in items
        ],
    }


@st.fragment(run_every=2)
def _batch_progress(worker: BatchWorker) -> None:
    """Live progress of a running batch with the time left, refreshed every two seconds without
    rerunning the page."""
    if not worker.is_alive():
        st.rerun()  # the whole page: export the result and show it
    # Read the checkpoint from disk: the worker's own copy may change while it is being read.
    saved = Checkpoint.load(worker.checkpoint.path) or worker.checkpoint
    scoring = worker.phase == "scoring"
    verb = "scored" if scoring else "written"
    keys, total = worker.keys, len(worker.keys)
    done = len(saved.scored(keys) if scoring else saved.generated(keys))
    # The last few rounds of blocks predict the rest best: a switch to another model changes the pace.
    recent = worker.seconds["scored" if scoring else "generated"][-worker.window:]
    seconds_left = sum(recent) / len(recent) * (total - done) if recent else None
    with st.container(border=True):
        model_note = f" · {worker.status['model']}" if not scoring and worker.status.get("model") else ""
        st.markdown(f":material/autorenew: **{'Scoring' if scoring else 'Writing'} {worker.plan.upload.name}** · "
                    f"{TASKS[worker.plan.task][0]}{model_note}")
        with st.container(horizontal=True):
            st.metric("Time left" + (" to score" if scoring else ""),
                      duration(seconds_left) if seconds_left is not None else "Timing the first block…", border=True)
            st.metric("Done around", (datetime.now() + timedelta(seconds=seconds_left)).strftime("%H:%M")
                      if seconds_left is not None else "Soon", border=True)
            st.metric(f"Blocks {verb}", f"{done} of {total}", border=True)
        st.progress(done / total)
        if note := worker.status.get("note"):
            st.caption(note)
        for line in worker.status.get("log", [])[-4:]:
            st.caption(line)
        if worker.stopping:
            st.button("Stopping after the blocks in progress…", disabled=True, icon=":material/hourglass_top:",
                      key=f"stopping:{worker.checkpoint.path}")
        elif st.button("Stop", icon=":material/stop_circle:", key=f"stop:{worker.checkpoint.path}",
                       help="Finishes the blocks being written, then stops. Every block written so far is saved "
                            "and exported; press Resume to continue."):
            worker.stop()
            st.rerun(scope="fragment")


def _finish_ended_batches() -> None:
    """Export every batch that has ended since the page last looked, with the settings it ran with,
    whatever the page is set to now, and say how it ended."""
    for path, worker in _own_batches().items():
        if worker.is_alive():
            continue
        running_batches().pop(path, None)
        plan, jobs, keys = worker.plan, worker.jobs, worker.keys
        checkpoint = Checkpoint.load(Path(path)) or worker.checkpoint
        saved, name = len(checkpoint.generated(keys)), plan.upload.name
        if worker.phase == "done":
            _export_batch(plan, jobs, checkpoint, partial=False)
            checkpoint.discard()
            notice = ("success", f"All {len(keys)} blocks of {name} are done and exported.")
        else:
            _export_batch(plan, jobs, checkpoint, partial=True)
            if worker.phase == "stopped":
                notice = ("warning", f"Stopped after {saved} of {len(keys)} blocks of {name}. They are saved and "
                                     "exported; press Resume to continue.")
            elif saved < len(keys):
                notice = ("error", f"Generation stopped at block {checkpoint.first_missing(keys)} of {len(keys)} of "
                                   f"{name}. The {saved} blocks written are saved and exported. Fix the cause, then "
                                   f"press Resume.\n\n{worker.error}")
            else:
                notice = ("error", f"Scoring stopped: {worker.error}. All {saved} blocks of {name} are generated, "
                                   "saved and exported. Press Resume to retry scoring, or turn scoring off "
                                   "and Resume to finish without scores.")
        st.session_state[NOTICE] = notice
        st.toast(notice[1].split(". ")[0].rstrip(".") + ".", icon={"success": ":material/check_circle:",
                                                        "warning": ":material/pause_circle:"}.get(
                                                            notice[0], ":material/error:"))


def _saved_progress(plan: Plan, jobs: list[dict], checkpoint: Checkpoint) -> None:
    """Offer a batch's saved progress, from an earlier run with these settings, to resume or discard."""
    keys = [job["key"] for job in jobs]
    saved, scored = len(checkpoint.generated(keys)), len(checkpoint.scored(keys))
    with st.container(border=True):
        st.markdown(f":material/save: **Saved progress for this document:** {saved} of {len(keys)} blocks "
                    "generated" + (f", {scored} scored" if plan.score else "")
                    + (f". Resume continues from block {checkpoint.first_missing(keys)}." if saved < len(keys)
                       else ". Resume finishes the batch."))
        if st.button("Discard and start over", icon=":material/delete:", type="tertiary"):
            checkpoint.discard()
            st.rerun()


def _model_order(plan: Plan, conn: Connection) -> list[str]:
    """The models to try in turn: the one the priorities picked, then the sidebar's."""
    return list(dict.fromkeys(name for name in (plan.model, conn.model, *conn.fallback_models) if name))


def _start_batch(plan: Plan, conn: Connection, jobs: list[dict], path: Path, checkpoint: Checkpoint | None,
                 system: str | None) -> None:
    keys = [job["key"] for job in jobs]
    if checkpoint is None:
        checkpoint = Checkpoint.start(
            path, {"source_file": plan.upload.name, "mode": plan.mode_tag, "blocks": len(jobs)}
        )
    by_key = {job["key"]: job for job in jobs}
    status = {"model": None, "note": None, "log": [], "released": False,
              "order": _model_order(plan, conn)}
    # Robustness mode drafts and scores each block on its own. A local server, as the main endpoint,
    # answers one request at a time anyway.
    group_size = 1 if plan.evasion else plan.group_size
    workers = 1 if plan.evasion or is_local(conn.base_url) else plan.parallel
    groups = [jobs[start:start + group_size] for start in range(0, len(jobs), group_size)]

    def group_tokens(group: list[dict]) -> int:
        """The most a local model may write for a group: each block's limit, and room for the JSON."""
        return sum(answer_token_limit(target_words(plan, job)) for job in group) + 40 * len(group)

    # One context window for the whole batch, big enough for its longest prompt, so the local model
    # loads once and keeps the shared document cached from block to block.
    local_context = max(
        [local_context_size(system, block_prompt(plan, job), group_tokens([job])) for job in jobs]
        + [local_context_size(system, group_prompt(plan, group), group_tokens(group))
           for group in groups if len(group) > 1]
    )
    local_model = conn.local_model
    if local_model and models.RAM_GB <= 8.5 and local_context > SMALL_MACHINE_LOCAL_CONTEXT:
        status["log"].append(
            f"Leaving out {local_model}: this document's prompts need a {local_context:,}-token window, more than "
            f"it can hold in {models.RAM_GB:.0f} GB. If the hosted models run out, the batch stops; press Resume "
            "once their quota is back."
        )
        local_model = ""
    shared = threading.Lock()  # groups written at the same time share the model order and the log

    # These run on the worker's threads, so they must not call Streamlit; they report through `status`.
    def write(prompt: str, max_tokens: int, **options) -> tuple[str, str, list[str]]:
        """Ask the models in turn. A model whose quota is spent for hours is dropped for the rest of the
        batch; when the ones left are only busy or rate-limited, wait and ask again."""
        for retry in range(BUSY_RETRIES + 1):
            spent: set[str] = set()
            try:
                return generate_with_fallback(conn.base_url, conn.api_key, list(status["order"]), prompt,
                                              plan.temperature, system, conn.local_url, local_model, max_tokens,
                                              local_context, spent, thinking=plan.thinking, **options)
            except APIError:
                left = [model for model in status["order"] if model not in spent]
                if retry == BUSY_RETRIES or not (left or local_model):
                    raise
            finally:
                if spent:
                    with shared:
                        status["log"].append(f"Dropping {', '.join(sorted(spent))} for this batch: quota spent.")
                        status["order"] = [model for model in status["order"] if model not in spent]
            for second in range(BUSY_WAIT * (retry + 1), 0, -1):
                if worker.stopping:
                    raise StopRequested
                status["note"] = (f"Every model is busy or rate-limited. Trying again in {duration(second)} "
                                  f"(try {retry + 2} of {BUSY_RETRIES + 1}).")
                time.sleep(1)
            status["note"] = None
        raise AssertionError("unreachable")

    def wrote_with(used_model: str, model_errors: list[str]) -> None:
        with shared:
            if used_model != status["model"]:
                status["log"].append(f"Writing with {used_model}" + (
                    f" after {len(model_errors)} model(s) failed. Last error: {model_errors[-1][:300]}"
                    if model_errors else "."
                ))
                status["model"] = used_model
            if used_model == local_model and not status["released"]:
                models.release()  # the local model needs the memory a loaded detector holds
                status["released"] = True
            # Later blocks start from the model that worked. The ones before it were only busy (spent ones
            # are dropped), so they move to the end rather than out.
            order = status["order"]
            if used_model in order:
                status["order"] = order[order.index(used_model) :] + order[: order.index(used_model)]

    def generate_block(job: dict) -> tuple[str, str]:
        prompt = block_prompt(plan, job)
        max_tokens = answer_token_limit(target_words(plan, job))
        if plan.evasion:
            candidates: dict[str, tuple[str, float | None]] = {}

            def draft(candidate_prompt: str) -> str:
                answer, candidate_model, errors = write(candidate_prompt, max_tokens)
                candidates[answer] = (candidate_model, errors)
                return answer

            def score_candidate(answer: str) -> float | None:
                if conn.local_model and candidates[answer][0] == local_model:
                    unload_local_model(conn.local_url, conn.local_model)
                return score_text(answer, plan.profile).get("ai_probability")

            status["note"] = f"Trying up to three drafts for {job['label']}…"
            result = generate_until_target(draft, score_candidate, f"{prompt}\n\n{PROFILE_INSTRUCTIONS}")
            answer = result.text
            used_model, model_errors = candidates.get(answer, (conn.model, []))
            status["note"] = None
        else:
            answer, used_model, model_errors = write(prompt, max_tokens)
        wrote_with(used_model, model_errors)
        return clean_essay(answer), used_model

    def generate_group(group: list[str]) -> Iterator[tuple[str, str, str]]:
        """Write a group of blocks in one request, then any block its reply leaves out or garbles on its own."""
        group_jobs = [by_key[key] for key in group]
        if len(group_jobs) > 1:
            reply, used_model, model_errors = write(group_prompt(plan, group_jobs), group_tokens(group_jobs),
                                                    json_schema=GROUP_SCHEMA, timeout=60 + 15 * len(group_jobs))
            wrote_with(used_model, model_errors)
            texts = parse_group_reply(reply, [f"B{number}" for number in range(1, len(group_jobs) + 1)])
            again = []
            for number, job in enumerate(group_jobs, 1):
                if text := clean_essay(texts.get(f"B{number}", "")):
                    yield job["key"], text, used_model
                else:
                    again.append(job)
            if again:
                with shared:
                    status["log"].append(f"{used_model} left out {len(again)} of {len(group_jobs)} blocks in a "
                                         "group; writing them one at a time.")
            group_jobs = again
        for job in group_jobs:
            yield job["key"], *generate_block(job)

    def before_scoring() -> None:
        if conn.local_model and any(checkpoint.blocks.get(key, {}).get("model") == conn.local_model for key in keys):
            status["note"] = f"Unloading {conn.local_model} to make room for the detector…"
            unload_local_model(conn.local_url, conn.local_model)
        status["note"] = f"Loading the {plan.profile} detector…"
        models.get_detector(plan.profile)
        status["note"] = None

    def score_document() -> None:
        # Answer boxes are mostly under the detector's 150-word minimum, so the regenerated text is
        # also scored as one document: the verdict a reader of the whole file would get.
        whole = "\n\n".join(checkpoint.blocks[key]["text"] for key in checkpoint.generated(keys))
        status["note"] = "Scoring the whole regenerated document…"
        checkpoint.meta["document_score"] = {**score_text(whole, plan.profile), "words": count_words(whole)}
        checkpoint.save()
        status["note"] = None

    awake = _keep_awake()

    def on_end() -> None:
        if awake:
            awake.terminate()
        saved, name = len(checkpoint.generated(keys)), plan.upload.name
        if worker.phase == "done":
            _notify(f"All {len(keys)} blocks of {name} are written and exported.")
        elif worker.phase == "stopped":
            _notify(f"Stopped after {saved} of {len(keys)} blocks of {name}.")
        else:
            _notify(f"Batch stopped at block {checkpoint.first_missing(keys) or saved + 1} of {len(keys)} of "
                    f"{name}: {str(worker.error)[:120]}")

    worker = BatchWorker(
        checkpoint, keys, generate_group,
        score=(lambda text: score_text(text, plan.profile)) if plan.score else None,
        before_scoring=before_scoring, finish=score_document if plan.score else None, status=status,
        on_end=on_end, group_size=group_size, workers=workers,
    )
    # Kept on the worker so that whichever page run sees it end can export it, even after the page's
    # settings have changed.
    worker.plan, worker.jobs = plan, jobs
    if not public():  # a public deploy's detector serves other visitors meanwhile
        models.release()  # the detector isn't needed until the blocks are written; it reloads to score them
    running_batches()[str(path)] = worker
    worker.start()


# -- single samples -------------------------------------------------------------------


def _generate_sample(plan: Plan, conn: Connection, system: str | None) -> None:
    """Generate one sample, score it, record it and export it."""
    upload, target_section = plan.upload, plan.target_section
    source_text = "\n\n".join(f"[{section.label}]\n{section.text}" for section in plan.context_sections)
    test_material = plan.prompt.strip()
    if source_text:
        test_material += "\n\nSource document material:\n" if test_material else "Source document material:\n"
        test_material += source_text
    if target_section:
        voice_rule = person_instruction(target_section.text, *(section.text for section in plan.context_sections))
        test_material += "\n\nPrevious answer in the selected section:\n" + target_section.text
        user_prompt = (
            "Create a new AI-generated red-team replacement for the selected answer section. "
            f"Aim for about {plan.words} words. "
            f"{voice_rule} "
            "Use the fixed question, prompt, other sections, and previous answer as context. "
            "Do not rewrite, answer, or "
            "repeat the fixed questions or any other sections. "
            "Return only the replacement answer text for the selected "
            "section. This is labelled AI-generated evaluation data, and the uploaded source file "
            "must remain unchanged.\n\n"
            f"Test material:\n{test_material}"
        )
    else:
        voice_rule = person_instruction(*(section.text for section in plan.sections))
        user_prompt = (
            f"Write an AI-generated red-team sample based on the test material below. Aim for about {plan.words} "
            "words. "
            f"{voice_rule} "
            "Return the sample only. This sample will be labelled as AI-generated evaluation data.\n\n"
            f"Test material:\n{test_material}"
        )
    model, model_errors = conn.model, []
    drafts: dict[str, str] = {}  # every draft written, with the model that wrote it
    reports: dict = {}  # the detector's report on each draft robustness mode scored

    def generate(prompt: str) -> str:
        nonlocal model, model_errors
        reply, model, model_errors = generate_with_fallback(
            conn.base_url, conn.api_key, _model_order(plan, conn), prompt,
            0.95 if plan.evasion else plan.temperature,
            system, conn.local_url, conn.local_model, answer_token_limit(plan.words), thinking=plan.thinking,
        )
        text = clean_essay(reply)
        drafts[text] = model
        return text

    def candidate_score(text: str) -> float | None:
        local = bool(conn.local_model) and model == conn.local_model
        if local:
            unload_local_model(conn.local_url, conn.local_model)  # the detector needs the memory it holds
        reports[text] = report = models.get_detector(plan.profile).analyze(text, sentences=False)
        if local:
            models.release()  # and the local model needs it back for the next draft
        return report.ai_probability

    def likelihood(text: str) -> float:
        report = reports.get(text)
        return report.ai_probability if report and report.ai_probability is not None else float("inf")

    try:
        if plan.evasion:
            with st.spinner("Generating and scoring up to 3 red-team drafts…"):
                try:
                    essay = generate_until_target(
                        generate, candidate_score, f"{user_prompt}\n\n{PROFILE_INSTRUCTIONS}").text
                except Exception as exc:  # a spent quota or a failed score; keep the drafts already written
                    if not drafts:
                        raise
                    essay = min(drafts, key=likelihood)
                    st.warning(f"Robustness mode stopped after {len(drafts)} draft(s), so the lowest-scoring one "
                               f"is kept: {exc}")
            model = drafts.get(essay, model)
        else:
            with st.spinner(f"Generating with {conn.model}…"):
                essay = generate(user_prompt)
    except APIError as exc:
        st.error(str(exc), icon=":material/error:")
        return

    score = reports.get(essay)
    if plan.score and score is None:
        try:
            if conn.local_model and model == conn.local_model:
                unload_local_model(conn.local_url, conn.local_model)  # the detector needs the memory it holds
            with st.spinner(f"Scoring with the {models.PROFILE_LABELS[plan.profile]} detector…"):
                score = models.get_detector(plan.profile).analyze(essay, sentences=False)
        except Exception as exc:  # Surface model/runtime failures without losing generated text.
            st.warning(f"The sample was generated, but detector scoring failed: {exc}")

    prompt_hash = hashlib.sha256(test_material.encode()).hexdigest()[:16]
    mode_tag = "regenerate_section" if target_section else "prompt"
    attack = (f"{mode_tag}+{plan.style_name}+{PROFILE_ID}" if plan.evasion
              else f"{mode_tag}+{plan.style_name}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    row = {
        "id": f"local-{timestamp}-{prompt_hash}-{model}-{attack}",
        "text": essay,
        "label": 1,
        "dataset": "local_redteam",
        "domain": "document_section" if target_section else "custom_prompt",
        "model": model,
        "attack": attack,
        "group": "ai",
        "group_key": f"prompt-{prompt_hash}",
        "prompt": test_material,
        "source_file": upload.name if upload else None,
        "target_section": target_section.label if target_section else None,
        "ai_probability": score.ai_probability
        if score and score.verdict not in ("insufficient_text", "not_prose")
        else None,
        "verdict": score.verdict if score else None,
        "calibrated": score.calibrated if score else None,
        "detector_profile": plan.profile if score else None,
    }
    reached = likelihood(essay) < EVASION_TARGET
    if plan.evasion:
        row |= {"evasion_mode": True, "evasion_profile": PROFILE_ID, "evasion_target": EVASION_TARGET,
                "evasion_attempts": len(drafts), "evasion_reached_target": reached, "evasion_kept": True}
    store = _store()
    store.output.parent.mkdir(parents=True, exist_ok=True)
    with store.output.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        # The drafts robustness mode passed over are labelled AI data too; they share the prompt's group_key,
        # so a split never puts one in training and another in test.
        for attempt, (draft, draft_model) in enumerate(drafts.items() if plan.evasion else (), 1):
            if draft != essay:
                report = reports.get(draft)
                output.write(json.dumps(row | {
                    "id": f"local-{timestamp}-{prompt_hash}-{draft_model}-{attack}-draft{attempt}",
                    "text": draft, "model": draft_model,
                    "ai_probability": report.ai_probability if report else None,
                    "verdict": report.verdict if report else None,
                    "calibrated": report.calibrated if report else None,
                    "detector_profile": plan.profile if report else None,
                    "evasion_reached_target": likelihood(draft) < EVASION_TARGET, "evasion_kept": False,
                }, ensure_ascii=False) + "\n")

    export_name = f"AI_REDTEAM_{mode_tag}_{timestamp}_{prompt_hash}.txt"
    export_path = store.exports / export_name
    attachment = (
        "Open Detector | AI-labelled red-team sample\n"
        f"Attack: {attack}\nModel: {model}\nSource: {upload.name if upload else 'prompt'}\n"
        f"Section: {target_section.label if target_section else 'new sample'}\n"
        f"Detector likelihood: {row['ai_probability'] if row['ai_probability'] is not None else 'not available'}\n\n"
        f"{essay}\n"
    )
    store.exports.mkdir(parents=True, exist_ok=True)
    export_path.write_text(attachment, encoding="utf-8")
    document_attachment = document_filename = document_path = None
    if plan.export_document and upload and target_section:
        try:
            document_attachment = replace_document_section(upload.name, upload.data, target_section, essay)
            document_filename = f"AI_REDTEAM_{mode_tag}_{timestamp}_{prompt_hash}{upload.suffix}"
            document_export_path = store.exports / document_filename
            document_export_path.write_bytes(document_attachment)
            document_path = _shown(document_export_path)
        except (ImportError, ValueError, IndexError, KeyError) as exc:
            st.warning(f"The answer was saved, but the document copy could not be exported: {exc}")
    st.session_state[RESULT] = {
        "essay": essay,
        "detect_text": essay,
        "row": row,
        "note": f"Written by {model} after {len(model_errors)} model(s) failed." if model_errors else None,
        "evasion_note": (f"Robustness mode wrote {len(drafts)} draft(s) and kept the lowest-scoring one, which "
                         + ("is under 5%." if reached else "is not under 5%.")) if plan.evasion else None,
        "score": (
            {
                "probability": score.ai_probability,
                "verdict": score.verdict,
                "calibrated": score.calibrated,
                "words": score.words,
            }
            if score
            else None
        ),
        "attachment": attachment,
        "filename": export_name,
        "path": _shown(export_path),
        "document_attachment": document_attachment,
        "document_filename": document_filename,
        "document_path": document_path,
    }


def _restore_last_result() -> None:
    """Show the most recent saved sample when the page opens in a new session."""
    store = _store()
    if RESULT in st.session_state or not store.output.is_file():
        return
    try:
        lines = store.output.read_text(encoding="utf-8").splitlines()
        saved_rows = [json.loads(line) for line in lines if line.strip()]
        if not saved_rows:
            return
        saved = saved_rows[-1]
        saved_id = saved.get("id", "latest")
        saved_hash = hashlib.sha256(saved_id.encode()).hexdigest()[:12]
        saved_name = f"AI_REDTEAM_previous_{saved_hash}.txt"
        saved_path = store.exports / saved_name
        saved_score = saved.get("ai_probability")
        if not saved_path.exists():
            store.exports.mkdir(parents=True, exist_ok=True)
            saved_path.write_text(
                "Open Detector | AI-labelled red-team sample\n"
                f"Attack: {saved.get('attack', 'red-team')}\n"
                f"Model: {saved.get('model', 'unknown')}\n"
                f"Source: {saved.get('source_file') or 'prompt'}\n"
                f"Section: {saved.get('target_section') or 'new sample'}\n"
                f"Detector likelihood: {saved_score if saved_score is not None else 'not available'}\n\n"
                f"{saved.get('text', '')}\n",
                encoding="utf-8",
            )
        st.session_state[RESULT] = {
            "essay": saved.get("text", ""),
            "detect_text": saved.get("text", ""),
            "row": saved,
            "score": (
                {
                    "probability": saved_score,
                    "verdict": saved.get("verdict"),
                    "calibrated": saved.get("calibrated"),
                    "words": count_words(saved.get("text", "")),
                }
                if saved.get("verdict")
                else None
            ),
            "attachment": saved_path.read_text(encoding="utf-8"),
            "filename": saved_name,
            "path": _shown(saved_path),
        }
    except (OSError, ValueError, KeyError):
        pass


# -- output ---------------------------------------------------------------------------


def _open_in_detect(text: str) -> None:
    st.session_state["keep:detect:source"] = detect.SOURCES[0]
    st.session_state["keep:detect:text"] = text
    st.session_state[detect.AUTORUN] = True
    st.switch_page(nav.detect_page())


def _show_result(result: dict, profile: str) -> None:
    row = result.get("row") or {}
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Latest sample**")
            if row.get("model"):
                st.badge(row["model"], icon=":material/smart_toy:", color="gray")
            if row.get("attack"):
                st.badge(row["attack"], icon=":material/label:", color="violet")
        if result.get("note"):
            st.caption(result["note"])
        if result.get("evasion_note"):
            st.caption(result["evasion_note"])

        score = result["score"]
        if score and score["verdict"] not in ("insufficient_text", "not_prose"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.metric(
                    result.get("score_label", "Detector") + " · AI likelihood", f"{score['probability']:.1%}",
                    help=f"{'Calibrated' if score['calibrated'] else 'Uncalibrated'} · "
                         f"{row.get('detector_profile') or profile} build",
                    width="content",
                )
                verdict_badge(score["verdict"])
        elif score:
            words = f" ({score['words']} words)" if score.get("words") else ""
            st.caption(f":material/info: The detector abstained: {VERDICTS[score['verdict']][0].lower()}{words}.")
        else:
            st.caption(":material/info: Not scored.")

        if result.get("batch_answers"):
            st.markdown(f"**{result.get('batch_summary') or 'Regenerated blocks'}** in the exported copy.")
            st.dataframe(
                [{"Block": item["label"], "AI likelihood": item["probability"],
                  "Verdict": VERDICTS[item["verdict"]][0] if item["verdict"] in VERDICTS else "Not scored"}
                 for item in result["batch_answers"]],
                hide_index=True, width="stretch",
                column_config={"AI likelihood": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1, color=style.tokens()["accent"])},
            )

        st.html(f'<div class="od-text od-sample">{html.escape(result["essay"])}</div>')
        with st.container(horizontal=True):
            st.download_button("Text file", result["attachment"], file_name=result["filename"], mime="text/plain",
                               icon=":material/download:", key="download_latest_redteam_sample")
            if result.get("document_attachment"):
                suffix = Path(result["document_filename"]).suffix
                st.download_button(f"Document copy ({suffix})", result["document_attachment"],
                                   file_name=result["document_filename"], mime=MIME_TYPES.get(suffix),
                                   icon=":material/download:", key="download_latest_redteam_document")
            if st.button("Analyze in Detect", icon=":material/search:", type="primary"):
                _open_in_detect(result.get("detect_text") or result["essay"])
        if public():
            st.caption("Recorded as AI text for this browser session only. Download what you want to keep: "
                       "reloading or closing the page forgets it.")
        else:
            saved = [Path(result["path"]).name] + ([result["document_filename"]] if result.get("document_path") else [])
            st.caption(f"Saved as {' and '.join(saved)} in {Path(result['path']).parent}, and recorded as AI text in "
                       f"{_shown(OUTPUT_PATH)}.")


@st.cache_data(show_spinner=False)
def _saved_rows(path: str, modified: float) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _history() -> None:
    """Every saved sample: how often each kind got past the detector."""
    output = _store().output
    if not output.is_file():
        st.info("No samples yet. Generated samples are listed here with their scores.", icon=":material/history:")
        return
    rows = _saved_rows(str(output), output.stat().st_mtime)
    scored = [row for row in rows if row.get("verdict") in ("likely_ai", "needs_review", "likely_human")]
    with st.container(horizontal=True):
        st.metric("Samples", f"{len(rows):,}", border=True)
        st.metric("Scored", f"{len(scored):,}", border=True)
        if scored:
            passed = sum(row["verdict"] == "likely_human" for row in scored) / len(scored)
            flagged = sum(row["verdict"] == "likely_ai" for row in scored) / len(scored)
            st.metric("Passed as human", f"{passed:.0%}", border=True,
                      help="Scored samples the detector called Likely human: its misses.")
            st.metric("Flagged as AI", f"{flagged:.0%}", border=True,
                      help="Scored samples the detector called Likely AI.")
    if scored:
        st.markdown("**By attack**")
        attacks: dict[str, list[dict]] = {}
        for row in scored:
            attacks.setdefault(row.get("attack") or "unknown", []).append(row)
        st.dataframe(
            sorted(
                ({"Attack": attack, "Scored": len(group),
                  "Passed as human": sum(r["verdict"] == "likely_human" for r in group) / len(group),
                  "Mean AI likelihood": sum(r.get("ai_probability") or 0 for r in group) / len(group)}
                 for attack, group in attacks.items()),
                key=lambda item: -item["Passed as human"],
            ),
            hide_index=True, width="stretch",
            column_config={
                "Passed as human": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                                   color=style.tokens()["accent"]),
                "Mean AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                                      color=style.tokens()["accent"]),
            },
        )
    st.markdown("**Latest samples**")
    latest = []
    for row in reversed(rows[-200:]):
        stamp = re.match(r"local-(\d{8}T\d{6}Z)", row.get("id", ""))
        latest.append({
            "Saved": datetime.strptime(stamp.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) if stamp else None,
            "Model": row.get("model"),
            "Attack": row.get("attack"),
            "Kind": row.get("domain"),
            "Words": count_words(row.get("text", "")),
            "AI likelihood": row.get("ai_probability"),
            "Verdict": VERDICTS[row["verdict"]][0] if row.get("verdict") in VERDICTS else "Not scored",
            "Text": row.get("text", "")[:300],
        })
    st.dataframe(
        latest, hide_index=True, width="stretch",
        column_config={
            "Saved": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
            "AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                             color=style.tokens()["accent"]),
            "Text": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("The samples from this browser session, labelled as AI text." if public()
               else f"All samples are in `{_shown(OUTPUT_PATH)}`, labelled as AI text.")


def render() -> None:
    load_env()
    profile = models.active_profile()
    conn = _connection()
    _restore_last_result()

    style.hero("Adversarial lab · " + ("public mode" if public() else "local"), "Red team",
               "Generate labelled AI samples and see how the detector scores them. Everything generated here is "
               "saved as AI-labelled evaluation data.",
               ["Your own Gemini key" if public() or packaged() else "Gemini, then a local fallback", "Style profiles",
                "Batch rewrites", "Scored per block"])
    _finish_ended_batches()
    running = [worker for worker in _own_batches().values() if worker.is_alive()]
    if running:
        # Generator mode: while a batch runs, the page shows only its progress. One batch runs at a time.
        st.markdown("A batch is running. Keep this tab open: reloading or closing it loses the results."
                    if public() else
                    "A batch is running. The detector is unloaded until the blocks are written, and the Mac stays "
                    "awake until it ends. You can close this tab: a notification says when it's done.")
        for worker in running:
            _batch_progress(worker)
        return
    st.caption(":material/shield: For detector evaluation only: don't submit generated work as your own.")

    generate_tab, saved_tab = st.tabs(["Generate", "Saved samples"])
    with generate_tab:
        setup, output = st.columns([1.1, 1], gap="large")
        with setup:
            plan = _setup(profile, conn)
            jobs = _batch_jobs(plan) if plan.batch else []
            path = _checkpoint_path(plan) if plan.batch else None
            checkpoint = Checkpoint.load(path) if plan.batch else None
            if notice := st.session_state.get(NOTICE):
                getattr(st, notice[0])(notice[1])
            if checkpoint:
                _saved_progress(plan, jobs, checkpoint)
                saved = len(checkpoint.generated([job["key"] for job in jobs]))
                label, icon = f"Resume · {saved} of {len(jobs)} blocks saved", ":material/play_arrow:"
            elif plan.rewrite_all_blocks:
                label, icon = f"Rewrite {len(jobs)} blocks", ":material/auto_awesome:"
            elif plan.fill_answer_boxes:
                label, icon = f"Fill {len(jobs)} answer boxes", ":material/auto_awesome:"
            else:
                label, icon = "Generate sample", ":material/auto_awesome:"
            clicked = st.button(label, type="primary", icon=icon, width="stretch")
            problem = _problem(plan, conn) if clicked else None
            if problem:
                st.error(problem)
            system = None
            if clicked and not problem and plan.use_profile:
                try:
                    system = style_instructions(load_style_profile(plan.style_path))
                except (OSError, ValueError) as exc:
                    st.error(f"Could not load the style profile: {exc}")
                    clicked = False

        with output:
            if clicked and not problem:
                st.session_state.pop(NOTICE, None)
                if plan.batch:
                    _start_batch(plan, conn, jobs, path, checkpoint, system)
                    st.rerun()
                _generate_sample(plan, conn, system)
            if result := st.session_state.get(RESULT):
                _show_result(result, profile)
            else:
                st.info("Set up a sample on the left and generate it. It appears here with its detector score.",
                        icon=":material/auto_awesome:")

    with saved_tab:
        _history()
