"""Local Streamlit interface for generating labelled detector red-team samples."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, UTC
from pathlib import Path

import streamlit as st

from eval.generate_essays import (
    DEFAULT_MODEL,
    GEMINI_API_URL,
    LOCAL_MODEL,
    LOCAL_URL,
    APIError,
    clean_essay,
    generate_with_fallback,
    load_style_profile,
    style_instructions,
    unload_local_model,
)
from eval.redteam_batch import BatchWorker, Checkpoint, batch_key
from opendetector import Detector, DetectorConfig
from opendetector.combiner import Combiner
from opendetector.engines import build_engine, free_memory
from opendetector.segment import count_words
from opendetector.documents import (
    extract_answer_slots,
    extract_document,
    replace_document_section,
)
from opendetector.env import load_env

ROOT = Path(__file__).resolve().parent
STYLE_PATH = ROOT / "data/styles/pasted_style_profile.json"
OUTPUT_PATH = ROOT / "data/essays/local_redteam.jsonl"
EXPORT_DIR = ROOT / "data/essays/exports"
BATCH_DIR = ROOT / "data/essays/batches"
PROFILES = ("slim", "full")
# The full detector needs about 3.5 GB. Next to a local model on an 8 GB machine that means heavy
# swapping, so small machines score with the slim build (Desklib alone, about 1.45 GB) by default.
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
PROFILE = os.environ.get("OPENDETECTOR_PROFILE") or ("slim" if RAM_GB <= 8.5 else "full")
DEFAULT_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)


def person_instruction(*samples: str) -> str:
    """Infer grammatical person from existing answer prose, ignoring question wording where possible."""
    text = " ".join(samples).casefold()
    counts = {
        "first person": len(re.findall(r"\b(?:i|me|my|mine|we|us|our|ours)\b", text)),
        "second person": len(re.findall(r"\b(?:you|your|yours|yourself|yourselves)\b", text)),
        "third person": len(re.findall(r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b", text)),
    }
    person, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        return (
            "Match the grammatical person used in the source text; do not add first- or "
            "second-person language if it is absent."
        )
    return f"Write in {person}, matching the source text. Keep that grammatical person consistent throughout."


st.set_page_config(page_title="Open Detector · Red-team generator", page_icon="🧪")
st.title("Red-team essay generator")
st.caption(
    "Generate labelled AI samples and measure how the local detector scores them. Generated text is saved as AI data."
)
st.warning(
    "Use this for detector evaluation. Do not submit generated work as your own. "
    "A style profile imitates writing patterns; it cannot reproduce a writer exactly."
)

load_env()

if "last_generated_result" not in st.session_state and OUTPUT_PATH.is_file():
    try:
        saved_rows = [json.loads(line) for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if saved_rows:
            saved = saved_rows[-1]
            saved_id = saved.get("id", "latest")
            saved_hash = hashlib.sha256(saved_id.encode()).hexdigest()[:12]
            saved_name = f"AI_REDTEAM_previous_{saved_hash}.txt"
            saved_path = EXPORT_DIR / saved_name
            saved_score = saved.get("ai_probability")
            if not saved_path.exists():
                EXPORT_DIR.mkdir(parents=True, exist_ok=True)
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
            st.session_state["last_generated_result"] = {
                "essay": saved.get("text", ""),
                "row": saved,
                "score": (
                    {
                        "probability": saved_score,
                        "verdict": saved.get("verdict"),
                        "calibrated": saved.get("calibrated"),
                        "words": None,
                    }
                    if saved.get("verdict")
                    else None
                ),
                "attachment": saved_path.read_text(encoding="utf-8"),
                "filename": saved_name,
                "path": str(saved_path.relative_to(ROOT)),
            }
    except (OSError, ValueError, KeyError):
        pass

with st.sidebar:
    st.header("Model connection")
    base_url = st.text_input("API URL (Gemini uses its Interactions API)", GEMINI_API_URL)
    model = st.text_input("Model", DEFAULT_MODEL)
    fallback_text = st.text_area(
        "Fallback text models, in order",
        value="\n".join(DEFAULT_FALLBACK_MODELS),
        height=150,
        help=(
            "Gemini uses these current text models after quota or temporary errors. "
            "Models unavailable to your account are skipped. Edit or clear this list "
            "for another OpenAI-compatible provider."
        ),
        key="fallback_models_interactions_api",
    )
    fallback_models = [line.strip() for line in fallback_text.replace(",", "\n").splitlines() if line.strip()]
    local_model = st.text_input(
        "Local fallback model (Ollama)", LOCAL_MODEL,
        help="Used after every model above fails. Runs on this computer with no key or quota; needs "
             "`ollama serve` running and the model fetched with `ollama pull`. Clear to turn it off.",
    ).strip()
    local_url = st.text_input("Local fallback URL", LOCAL_URL).strip()
    api_key_env = st.text_input("API key environment variable", "GEMINI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if api_key:
        st.caption(f"Using key from {api_key_env} or the project .env file.")
    else:
        st.caption(f"No {api_key_env} found. Local endpoints can work without a key.")

prompt = st.text_area(
    "Test prompt or instructions", height=140, placeholder="Describe what the red-team sample should cover."
)
source_upload = st.file_uploader("Optional source material (.md, .docx, or .pptx)", type=("md", "docx", "pptx"))
source_sections = []
answer_slots = []
if source_upload:
    try:
        source_sections = extract_document(source_upload.name, source_upload.getvalue())
        answer_slots = extract_answer_slots(source_upload.name, source_upload.getvalue())
        st.caption(f"Extracted {len(source_sections)} sections from {source_upload.name}.")
        with st.expander("Review extracted source text"):
            for section in source_sections:
                st.markdown(f"**{section.label}**")
                st.text(section.text)
    except (ImportError, ValueError) as exc:
        st.error(f"Could not read {source_upload.name}: {exc}")
        st.stop()

fill_answer_boxes = False
if source_upload:
    fill_answer_boxes = st.checkbox(
        "Generate new AI answers in the right-hand boxes and export a filled copy",
        value=False,
        disabled=not answer_slots,
        help=(
            "Uses each left-hand question and the document's existing text as context. "
            "Replaces text only in paired right-hand answer boxes, including Markdown table rows."
        ),
    )
    if answer_slots:
        st.caption(
            f"Found {len(answer_slots)} question/answer box pairs. Existing answers may be "
            "blank or filled; the right-hand box will be replaced."
        )
        with st.expander("Review detected question and answer boxes"):
            for slot in answer_slots:
                st.markdown(f"**{slot.label}**")
                st.write(f"Question: {slot.question}")
                st.write(f"Current answer: {slot.previous_answer or '(blank)'}")
    else:
        st.caption("No paired question/answer tables or left/right PowerPoint text boxes were detected in this file.")

markdown_targets = [
    section
    for section in source_sections
    if section.kind == "markdown_block"
    and "?" not in section.text
    and not section.text.strip().casefold().startswith(("question:", "prompt:"))
]
if source_upload and Path(source_upload.name).suffix.lower() == ".md" and markdown_targets:
    generation_modes = [
        "Regenerate entire Markdown file (all text blocks)",
        "New red-team sample",
        "Regenerate one section",
    ]
else:
    generation_modes = ["New red-team sample", "Regenerate one section"]

generation_mode = st.radio(
    "Generation mode",
    generation_modes,
    horizontal=True,
    disabled=not bool(source_sections),
    help="Regeneration creates a separate replacement sample. It never edits the uploaded file.",
)
target_section = None
context_sections = source_sections
export_document = False
rewrite_all_markdown = generation_mode == "Regenerate entire Markdown file (all text blocks)"
if generation_mode == "Regenerate one section" and source_sections and not fill_answer_boxes:
    section_labels = [section.label for section in source_sections]
    target_label = st.selectbox("Section to regenerate", section_labels)
    target_section = next(section for section in source_sections if section.label == target_label)
    other_sections = [section for section in source_sections if section.label != target_label]
    context_labels = st.multiselect(
        "Other sections to provide as context",
        [section.label for section in other_sections],
        default=[section.label for section in other_sections],
        help="Questions and earlier answers can help the model keep the replacement consistent.",
    )
    context_sections = [section for section in other_sections if section.label in context_labels]
    st.info("The selected section is treated as the previous AI answer. The uploaded file remains unchanged.")
    export_document = st.checkbox(
        "Also export a copy of the uploaded file with this section replaced",
        value=False,
        help="Exports a new AI-labelled Markdown, DOCX, or PPTX copy. The uploaded original is never modified.",
    )
if rewrite_all_markdown:
    st.caption(
        f"Will regenerate all {len(markdown_targets)} Markdown text blocks in one pass each. "
        "Headings, question blocks, and table cells stay unchanged unless you also tick the answer-box option."
    )
    export_document = True

words = st.slider("Approximate length", min_value=150, max_value=1500, value=350, step=50)
batch_regeneration = fill_answer_boxes or rewrite_all_markdown
answer_words = st.slider(
    "Approximate words per answer box", min_value=20, max_value=500, value=100, step=10, disabled=not batch_regeneration
)
match_existing_lengths = st.checkbox(
    "Match the length of existing text blocks",
    value=True,
    disabled=not batch_regeneration,
)
temperature = st.slider("Creativity", 0.0, 1.5, 0.8, 0.1)
use_profile = st.checkbox("Use the pasted writer style profile", value=True)
score_with_detector = st.checkbox("Score the sample with the local detector", value=True)
detector_profile = st.selectbox(
    "Detector for scoring",
    PROFILES,
    index=PROFILES.index(PROFILE) if PROFILE in PROFILES else 1,
    disabled=not score_with_detector,
    help=(
        "slim is Desklib alone (about 1.45 GB); full adds GLYPH and Binoculars (about 3.5 GB). "
        f"This computer has {RAM_GB:.0f} GB, and a local text model needs about 3 GB more."
    ),
)


def build_detector(profile: str) -> Detector:
    config = DetectorConfig.for_profile(profile)
    engines = [
        build_engine(name, config)
        for name, model_id in (("glyph", config.glyph_model), ("desklib", config.desklib_model))
        if model_id
    ]
    bino = build_engine("bino", config) if config.binoculars_observer else None
    combiner = Combiner.load(config.combiner_path) if config.combiner_path.exists() else None
    return Detector(engines, bino, combiner, config)


@st.cache_resource
def detector_cache() -> dict:
    """The loaded detector, shared by every session and by batch workers. Only one profile is kept
    at a time: two would not fit next to a local text model on an 8 GB machine."""
    return {"lock": threading.Lock(), "profile": None, "detector": None}


@st.cache_resource
def running_batches() -> dict:
    """Batch workers by checkpoint file, shared by every session and kept across reruns."""
    return {}


DETECTORS = detector_cache()


def get_detector(profile: str) -> Detector:
    """The detector for `profile`, loading it (and dropping any other) on first use. Safe to call from
    a batch worker's thread: it makes no Streamlit calls."""
    with DETECTORS["lock"]:
        if DETECTORS["profile"] != profile:
            DETECTORS["detector"] = DETECTORS["profile"] = None
            free_memory()
            DETECTORS["detector"], DETECTORS["profile"] = build_detector(profile), profile
        return DETECTORS["detector"]


def release_detector() -> None:
    """Drop the loaded detector, so a local text model has the memory to itself."""
    with DETECTORS["lock"]:
        DETECTORS["detector"] = DETECTORS["profile"] = None
    free_memory()


def score_text(text: str, profile: str) -> dict:
    """A block's score record. Text under the detector's minimum is not run through it."""
    if count_words(text) < DetectorConfig.for_profile(profile).min_words:
        result = {"ai_probability": None, "verdict": "insufficient_text", "calibrated": None}
    else:
        analysis = get_detector(profile).analyze(text, sentences=False)
        result = {
            "ai_probability": analysis.ai_probability
            if analysis.verdict not in ("insufficient_text", "not_prose")
            else None,
            "verdict": analysis.verdict,
            "calibrated": analysis.calibrated,
        }
    return {**result, "detector_profile": profile}


def duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


style_name = STYLE_PATH.stem if use_profile else "plain"
mode_tag = "full_markdown" if rewrite_all_markdown else "filled_answers"
jobs = []
checkpoint = None
worker = None
if batch_regeneration:
    if fill_answer_boxes:
        jobs.extend(
            {
                "kind": "answer_box",
                "label": slot.label,
                "question": slot.question,
                "previous": slot.previous_answer,
                "target": slot.target,
            }
            for slot in answer_slots
        )
    if rewrite_all_markdown:
        jobs.extend(
            {
                "kind": "markdown_block",
                "label": section.label,
                "question": "",
                "previous": section.text,
                "target": section,
            }
            for section in markdown_targets
        )
    # Labels can repeat (two shapes with the same name), so a block's position is part of its key.
    for index, job in enumerate(jobs, 1):
        job["key"] = f"{index:03d} {job['kind']} {job['label']}"
    source_hash = hashlib.sha256(source_upload.getvalue()).hexdigest()[:16]
    context_text = "\n\n".join(f"[{section.label}]\n{section.text}" for section in source_sections)
    checkpoint_path = BATCH_DIR / (
        batch_key(source_hash, fill_answer_boxes, rewrite_all_markdown, prompt.strip(), use_profile,
                  answer_words, match_existing_lengths) + ".json"
    )
    checkpoint = Checkpoint.load(checkpoint_path)
    worker = running_batches().get(str(checkpoint_path))
keys = [job["key"] for job in jobs]


def export_batch(checkpoint: Checkpoint, partial: bool) -> None:
    """Export every generated block: a copy of the document with those blocks replaced (in a partial
    export the rest stay as uploaded) and a text file of the answers. Blocks that are final, scored or
    with scoring off, are appended to the evaluation records, once each."""
    generated = set(checkpoint.generated(keys))
    items = [{**job, **checkpoint.blocks[job["key"]]} for job in jobs if job["key"] in generated]
    if not items:
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{mode_tag}_partial" if partial else mode_tag
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document_bytes = document_name = document_path = None
    try:
        document_bytes = source_upload.getvalue()
        for item in sorted(items, key=lambda value: value["target"].locator[0], reverse=True):
            document_bytes = replace_document_section(source_upload.name, document_bytes, item["target"], item["text"])
        document_name = f"AI_REDTEAM_{tag}_{timestamp}_{source_hash}{Path(source_upload.name).suffix.lower()}"
        (EXPORT_DIR / document_name).write_bytes(document_bytes)
        document_path = str((EXPORT_DIR / document_name).relative_to(ROOT))
    except (ImportError, ValueError, IndexError, KeyError) as exc:
        document_bytes = document_name = None
        st.warning(f"The answers are saved, but the document copy could not be exported: {exc}")

    attack = f"{mode_tag}+{style_name}"
    rows = []
    for item in items:
        if item.get("recorded") or (score_with_detector and "score" not in item):
            continue
        question_material = f"Target: {item['label']}\n"
        if item["question"]:
            question_material += f"Question:\n{item['question']}\n"
        question_material += (
            f"Previous text:\n{item['previous']}\n\nAdditional instructions:\n"
            f"{prompt.strip()}\n\nDocument context:\n{context_text}"
        )
        record_hash = hashlib.sha256(item["key"].encode()).hexdigest()[:8]
        rows.append({
            "id": f"local-{checkpoint.meta['created']}-{source_hash}-{record_hash}-{item['model']}-{attack}",
            "text": item["text"],
            "label": 1,
            "dataset": "local_redteam",
            "domain": "document_answer_box" if item["kind"] == "answer_box" else "markdown_text_block",
            "model": item["model"],
            "attack": attack,
            "group": "ai",
            "group_key": f"document-{source_hash}",
            "prompt": question_material,
            "source_file": source_upload.name,
            "target_section": item["label"],
            "detector_profile": None,
            **(item.get("score") or {}),
        })
        checkpoint.blocks[item["key"]]["recorded"] = True
    if rows:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("a", encoding="utf-8") as output:
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
        f"Source document: {source_upload.name}\n"
        f"Regenerated blocks: {len(items)} of {len(jobs)}\n\n" + "\n\n".join(answers_text) + "\n"
    )
    (EXPORT_DIR / text_name).write_text(text_attachment, encoding="utf-8")
    noun = "Markdown text blocks" if rewrite_all_markdown else "answer boxes"
    st.session_state["last_generated_result"] = {
        "essay": "\n\n".join(answers_text),
        "row": rows[0] if rows else {},
        "score": None,
        "attachment": text_attachment,
        "filename": text_name,
        "path": str((EXPORT_DIR / text_name).relative_to(ROOT)),
        "document_attachment": document_bytes,
        "document_filename": document_name,
        "document_path": document_path,
        "batch_summary": (
            f"{'Regenerated' if rewrite_all_markdown else 'Filled'} {len(items)} of {len(jobs)} {noun}"
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
def show_batch_progress(worker: BatchWorker, total: int) -> None:
    """Live progress of a running batch, refreshed every two seconds without rerunning the page."""
    if not worker.is_alive():
        st.rerun()  # the whole page: export the result and show it
    # Read the checkpoint from disk: the worker's own copy may change while it is being read.
    saved = Checkpoint.load(worker.checkpoint.path) or worker.checkpoint
    scoring = worker.phase == "scoring"
    verb = "scored" if scoring else "generated"
    done = len(saved.scored(keys) if scoring else saved.generated(keys))
    times, left = worker.seconds[verb], total - done
    eta = f" · about {duration(sum(times) / len(times) * left)} left" if times and left else ""
    with st.container(border=True):
        model_note = f" · {worker.status['model']}" if not scoring and worker.status.get("model") else ""
        st.markdown(f"**{'Scoring' if scoring else 'Generating'} {total} blocks**{model_note}")
        st.progress(done / total, text=f"{done} of {total} blocks {verb}{eta}")
        if note := worker.status.get("note"):
            st.caption(note)
        for line in worker.status.get("log", [])[-4:]:
            st.caption(line)
        if worker.stopping:
            st.button("Stopping after the block in progress…", disabled=True)
        elif st.button("Stop generating", help="Finishes the block in progress, then stops. Every block "
                       "written so far is saved and exported; press Resume to continue."):
            worker.stop()
            st.rerun(scope="fragment")


if batch_regeneration:
    if worker and not worker.is_alive():
        # The batch has ended: export what it wrote, and say how it ended.
        running_batches().pop(str(checkpoint_path), None)
        checkpoint = Checkpoint.load(checkpoint_path) or worker.checkpoint
        saved = len(checkpoint.generated(keys))
        if worker.phase == "done":
            export_batch(checkpoint, partial=False)
            checkpoint.discard()
            checkpoint = None
            st.success(f"All {len(keys)} blocks are done and exported below.")
        else:
            export_batch(checkpoint, partial=True)
            if worker.phase == "stopped":
                st.warning(f"Stopped after {saved} of {len(keys)} blocks. They are saved and exported below; "
                           "press Resume to continue.")
            elif saved < len(keys):
                st.error(f"Generation stopped at block {saved + 1} of {len(keys)}. The {saved} blocks written "
                         f"before it are saved and exported below. Fix the cause, then press Resume.\n\n"
                         f"{worker.error}")
            else:
                st.error(f"Scoring stopped: {worker.error}. All {saved} blocks are generated, saved and "
                         "exported below. Press Resume to retry scoring, or turn scoring off and Resume to "
                         "finish without scores.")
        worker = None
    if worker:
        show_batch_progress(worker, len(keys))
    elif checkpoint:
        saved, scored = len(checkpoint.generated(keys)), len(checkpoint.scored(keys))
        st.info(
            f"Saved progress for this document: {saved} of {len(keys)} blocks generated"
            + (f", {scored} scored" if score_with_detector else "")
            + (f". Resume continues from block {saved + 1}." if saved < len(keys) else ". Resume finishes the batch.")
        )
        if st.button("Discard saved progress and start over"):
            checkpoint.discard()
            st.rerun()

button_label = (
    f"Resume ({len(checkpoint.generated(keys))} of {len(keys)} blocks saved)"
    if checkpoint
    else "Regenerate all and export Markdown"
    if rewrite_all_markdown
    else "Generate answers and export filled document"
    if fill_answer_boxes
    else "Generate labelled red-team sample"
)
if st.button(button_label, type="primary", disabled=worker is not None):
    if not prompt.strip() and not source_sections:
        st.error("Enter a test prompt or upload a Markdown, Word, or PowerPoint document first.")
        st.stop()
    if generation_mode == "Regenerate one section" and target_section is None and not fill_answer_boxes:
        st.error("Choose a document section to regenerate.")
        st.stop()
    if not api_key and not base_url.strip().startswith(("http://localhost", "http://127.0.0.1")):
        st.error(f"Set {api_key_env} in the project .env file or use a local model endpoint.")
        st.stop()

    system = None
    if use_profile:
        try:
            system = style_instructions(load_style_profile(STYLE_PATH))
        except (OSError, ValueError) as exc:
            st.error(f"Could not load style profile: {exc}")
            st.stop()

    if batch_regeneration:
        if checkpoint is None:
            checkpoint = Checkpoint.start(
                checkpoint_path, {"source_file": source_upload.name, "mode": mode_tag, "blocks": len(jobs)}
            )
        by_key = {job["key"]: job for job in jobs}
        status = {"model": None, "note": None, "log": [], "released": False,
                  "order": list(dict.fromkeys([model, *fallback_models]))}

        # These run on the worker's thread, so they must not call Streamlit; they report through `status`.
        def generate_block(key: str) -> tuple[str, str]:
            job = by_key[key]
            voice_samples = [job["previous"]]
            if job["kind"] == "answer_box":
                voice_samples.extend(slot.previous_answer for slot in answer_slots if slot.previous_answer)
            else:
                voice_samples.extend(section.text for section in markdown_targets if section.text != job["previous"])
            voice_rule = person_instruction(*voice_samples)
            target_words = (
                count_words(job["previous"]) if match_existing_lengths and job["previous"].strip() else answer_words
            )
            if job["kind"] == "answer_box":
                generation_prompt = (
                    f"Generate a new AI red-team answer for the question in {job['label']}. "
                    f"Aim for about {target_words} words. "
                    f"{voice_rule} "
                    "Use the document's other questions and previous answers as context. "
                    "Replace the existing answer if one "
                    "is present. Do not change, repeat, or answer any other question. Return only the new answer text. "
                    "This answer is AI-generated red-team evaluation data.\n\n"
                    f"Additional instructions:\n{prompt.strip() or '(none)'}\n\n"
                    f"Question to answer:\n{job['question']}\n\n"
                    f"Previous answer in this right-hand box (may be blank):\n{job['previous'] or '(blank)'}\n\n"
                    f"Other document context:\n{context_text}"
                )
            else:
                generation_prompt = (
                    f"Rewrite only the existing Markdown text block labelled {job['label']}. "
                    f"Aim for about {target_words} words. "
                    f"{voice_rule} "
                    "Keep its meaning and role in the document, using the surrounding document as context. "
                    "Return only the "
                    "replacement for this block. Do not output headings, questions, table rows, or any other block. "
                    "This replacement is AI-generated red-team evaluation data.\n\n"
                    f"Additional instructions:\n{prompt.strip() or '(none)'}\n\n"
                    f"Existing text to regenerate:\n{job['previous']}\n\n"
                    f"Full document context:\n{context_text}"
                )
            answer, used_model, model_errors = generate_with_fallback(
                base_url, api_key, status["order"], generation_prompt, temperature, system, local_url, local_model,
            )
            if used_model != status["model"]:
                status["log"].append(f"Writing with {used_model}" + (
                    f" after {len(model_errors)} model(s) failed. Last error: {model_errors[-1][:300]}"
                    if model_errors else "."
                ))
                status["model"] = used_model
            if used_model == local_model and not status["released"]:
                release_detector()  # the local model needs the memory a loaded detector holds
                status["released"] = True
            # Later blocks start from the model that worked; once that is the local model, the
            # hosted ones are spent and every later block goes straight to it.
            order = status["order"]
            status["order"] = order[order.index(used_model) :] if used_model in order else []
            return clean_essay(answer), used_model

        def before_scoring() -> None:
            if local_model and any(checkpoint.blocks.get(key, {}).get("model") == local_model for key in keys):
                status["note"] = f"Unloading {local_model} to make room for the detector…"
                unload_local_model(local_url, local_model)
            status["note"] = f"Loading the {detector_profile} detector…"
            get_detector(detector_profile)
            status["note"] = None

        worker = BatchWorker(
            checkpoint, keys, generate_block,
            score=(lambda text: score_text(text, detector_profile)) if score_with_detector else None,
            before_scoring=before_scoring, status=status,
        )
        running_batches()[str(checkpoint_path)] = worker
        worker.start()
        st.rerun()

    source_text = "\n\n".join(f"[{section.label}]\n{section.text}" for section in context_sections)
    test_material = prompt.strip()
    if source_text:
        test_material += "\n\nSource document material:\n" if test_material else "Source document material:\n"
        test_material += source_text
    if generation_mode == "Regenerate one section" and target_section:
        voice_rule = person_instruction(target_section.text, *(section.text for section in context_sections))
        test_material += "\n\nPrevious answer in the selected section:\n" + target_section.text
        user_prompt = (
            "Create a new AI-generated red-team replacement for the selected answer section. "
            f"Aim for about {words} words. "
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
        voice_rule = person_instruction(*(section.text for section in source_sections))
        user_prompt = (
            f"Write an AI-generated red-team sample based on the test material below. Aim for about {words} words. "
            f"{voice_rule} "
            "Return the sample only. This sample will be labelled as AI-generated evaluation data.\n\n"
            f"Test material:\n{test_material}"
        )
    try:
        with st.spinner(f"Generating with {model}…"):
            essay, model, model_errors = generate_with_fallback(
                base_url, api_key, [model, *fallback_models], user_prompt, temperature, system,
                local_url, local_model,
            )
    except APIError as exc:
        st.error(str(exc))
        st.stop()
    essay = clean_essay(essay)
    if model_errors:
        st.info(f"Generated with {model} after earlier model(s) failed.")

    score = None
    if score_with_detector:
        try:
            if local_model and model == local_model:
                unload_local_model(local_url, local_model)  # the detector needs the memory it holds
            with st.spinner(f"Scoring with the local {detector_profile} detector…"):
                score = get_detector(detector_profile).analyze(essay, sentences=False)
        except Exception as exc:  # Surface model/runtime failures without losing generated text.
            st.warning(f"The essay was generated, but detector scoring failed: {exc}")

    prompt_hash = hashlib.sha256(test_material.encode()).hexdigest()[:16]
    mode_tag = "regenerate_section" if target_section else "prompt"
    attack = f"{mode_tag}+{style_name}"
    row = {
        "id": f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{prompt_hash}-{model}-{attack}",
        "text": essay,
        "label": 1,
        "dataset": "local_redteam",
        "domain": "document_section" if target_section else "custom_prompt",
        "model": model,
        "attack": attack,
        "group": "ai",
        "group_key": f"prompt-{prompt_hash}",
        "prompt": test_material,
        "source_file": source_upload.name if source_upload else None,
        "target_section": target_section.label if target_section else None,
        "ai_probability": score.ai_probability
        if score and score.verdict not in ("insufficient_text", "not_prose")
        else None,
        "verdict": score.verdict if score else None,
        "calibrated": score.calibrated if score else None,
        "detector_profile": detector_profile if score else None,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    export_name = f"AI_REDTEAM_{mode_tag}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{prompt_hash}.txt"
    export_path = EXPORT_DIR / export_name
    attachment = (
        "Open Detector | AI-labelled red-team sample\n"
        f"Attack: {attack}\nModel: {model}\nSource: {source_upload.name if source_upload else 'prompt'}\n"
        f"Section: {target_section.label if target_section else 'new sample'}\n"
        f"Detector likelihood: {row['ai_probability'] if row['ai_probability'] is not None else 'not available'}\n\n"
        f"{essay}\n"
    )
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path.write_text(attachment, encoding="utf-8")
    document_attachment = None
    document_filename = None
    document_path = None
    if export_document and source_upload and target_section:
        try:
            document_attachment = replace_document_section(
                source_upload.name, source_upload.getvalue(), target_section, essay
            )
            extension = Path(source_upload.name).suffix.lower()
            document_filename = (
                f"AI_REDTEAM_{mode_tag}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{prompt_hash}{extension}"
            )
            document_export_path = EXPORT_DIR / document_filename
            document_export_path.write_bytes(document_attachment)
            document_path = str(document_export_path.relative_to(ROOT))
        except (ImportError, ValueError, IndexError, KeyError) as exc:
            st.warning(f"The answer was saved, but the document copy could not be exported: {exc}")
    st.session_state["last_generated_result"] = {
        "essay": essay,
        "row": row,
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
        "path": str(export_path.relative_to(ROOT)),
        "document_attachment": document_attachment,
        "document_filename": document_filename,
        "document_path": document_path,
    }

if result := st.session_state.get("last_generated_result"):
    st.subheader("Generated file")
    st.success("Your AI-labelled red-team sample is ready to download.")
    st.caption(f"Saved at `{result['path']}` · Evaluation record: `{OUTPUT_PATH.relative_to(ROOT)}`")
    score_result = result["score"]
    if score_result:
        if score_result["verdict"] not in ("insufficient_text", "not_prose"):
            st.metric(
                "Detector AI likelihood",
                f"{score_result['probability']:.1%}",
                help=f"{score_result['verdict'].replace('_', ' ').title()} · "
                f"{'calibrated' if score_result['calibrated'] else 'uncalibrated'} · "
                f"{result['row'].get('detector_profile') or detector_profile} profile",
            )
        else:
            st.info(f"Detector abstained: {score_result['verdict'].replace('_', ' ')} ({score_result['words']} words).")
    st.download_button(
        "Download generated answer (.txt)",
        result["attachment"],
        file_name=result["filename"],
        mime="text/plain",
        type="primary",
        key="download_latest_redteam_sample",
    )
    if result.get("batch_answers"):
        batch_summary = result.get("batch_summary") or f"Regenerated {len(result['batch_answers'])} text blocks"
        st.markdown(f"**{batch_summary}** in the exported copy.")
        st.dataframe(
            [
                {
                    "answer box": item["label"],
                    "detector likelihood": (
                        f"{item['probability']:.1%}" if item["probability"] is not None else "Not scored"
                    ),
                    "verdict": (item["verdict"].replace("_", " ") if item["verdict"] else "Not scored"),
                }
                for item in result["batch_answers"]
            ],
            hide_index=True,
            width="stretch",
        )
    if result.get("document_attachment"):
        st.download_button(
            "Download regenerated document (.md/.docx/.pptx)",
            result["document_attachment"],
            file_name=result["document_filename"],
            mime=(
                "text/markdown"
                if result["document_filename"].endswith(".md")
                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if result["document_filename"].endswith(".docx")
                else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ),
            type="primary",
            key="download_latest_redteam_document",
        )
        st.caption(f"Document copy saved at `{result['document_path']}`. The uploaded original is unchanged.")
    with st.expander("Preview generated answer"):
        st.text_area("AI-labelled red-team sample", result["essay"], height=300, key="latest_redteam_preview")
