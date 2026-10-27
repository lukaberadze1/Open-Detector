"""The Red team page: generate labelled AI samples and see how the detector scores them.

Everything it writes is AI-labelled evaluation data. Each sample is appended to
data/essays/local_redteam.jsonl with label 1 and exported under data/essays/exports/. Batches, which
regenerate every answer box or text block of an uploaded document, run on a worker thread and save
each block as they go (eval/redteam_batch.py), so reruns, page switches and closed tabs don't end them.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    local_model_problem,
    style_instructions,
    unload_local_model,
)
from eval.redteam_batch import BatchWorker, Checkpoint, batch_key
from opendetector import DetectorConfig
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

from . import detect, models, nav
from .components import VERDICTS, Upload, keep, persistent_upload, verdict_badge

ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = ROOT / "data/styles/alexandria_profile.json"
OUTPUT_PATH = ROOT / "data/essays/local_redteam.jsonl"
EXPORT_DIR = ROOT / "data/essays/exports"
BATCH_DIR = ROOT / "data/essays/batches"
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
NOTICE = "redteam:notice"  # how the last batch ended, shown until the next run starts


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


# Checked on every rerun to show in the sidebar, so the answer is kept for a few seconds.
local_fallback_problem = st.cache_data(ttl=15, show_spinner=False)(local_model_problem)


@st.cache_resource
def running_batches() -> dict:
    """Batch workers by checkpoint file, shared by every session and kept across reruns."""
    return {}


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
    use_profile: bool = True
    score: bool = True

    @property
    def batch(self) -> bool:
        return self.fill_answer_boxes or self.rewrite_all_blocks

    @property
    def style_name(self) -> str:
        return STYLE_PATH.stem if self.use_profile else "plain"

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
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


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
                "Local fallback model (Ollama)", key=keep("keep:redteam:local_model", LOCAL_MODEL),
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
                st.caption(f":material/key: Using the key in {api_key_env} or the project .env file.")
            else:
                st.caption(f":material/key_off: No {api_key_env} found. Local endpoints work without a key.")
            if local_model:
                if problem := local_fallback_problem(local_url, local_model):
                    st.caption(f":orange[:material/cloud_off: The local fallback can't run: {problem}]")
                else:
                    st.caption(f":material/check_circle: Local fallback ready: {local_model}.")
    return Connection(base_url, model, fallback_models, local_model, local_url, api_key_env, api_key)


def _setup(profile: str) -> Plan:
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
        st.caption(f"Rewrites {len(blocks)} text blocks, one model call each. Headings, titles, questions and "
                   "tables stay as they are" + (" unless you fill the answer boxes too." if slots else "."))
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
    style_found = STYLE_PATH.is_file()
    use_style = st.toggle(
        "Write in the Alexandria style", key=keep("keep:redteam:style", True), disabled=not style_found,
        help=f"Sends the style profile {STYLE_PATH.name} as the system prompt. A style profile imitates writing "
             "patterns; it can't reproduce a writer exactly.",
    )
    plan.use_profile = style_found and use_style
    if not style_found:
        st.caption(f"No style profile at data/styles/{STYLE_PATH.name}, so samples use the model's own style.")
    plan.score = st.toggle("Score with the detector", key=keep("keep:redteam:score", True),
                           help=f"Uses the {models.PROFILE_LABELS[profile]} build chosen in the sidebar. The "
                                "first score loads its models.")
    return plan


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
    if not conn.api_key and not conn.base_url.strip().startswith(("http://localhost", "http://127.0.0.1")):
        return f"Set {conn.api_key_env} in the project .env file or use a local model endpoint."
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
    return BATCH_DIR / (
        batch_key(plan.source_hash, plan.fill_answer_boxes, plan.rewrite_all_blocks, plan.prompt.strip(),
                  plan.use_profile, plan.answer_words, plan.match_existing_lengths) + ".json"
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
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    document_bytes = document_name = document_path = None
    try:
        document_bytes = upload.data
        for item in sorted(items, key=lambda value: value["target"].locator[0], reverse=True):
            document_bytes = replace_document_section(upload.name, document_bytes, item["target"], item["text"])
        document_name = f"AI_REDTEAM_{tag}_{timestamp}_{source_hash}{upload.suffix}"
        (EXPORT_DIR / document_name).write_bytes(document_bytes)
        document_path = str((EXPORT_DIR / document_name).relative_to(ROOT))
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
        f"Source document: {upload.name}\n"
        f"Regenerated blocks: {len(items)} of {len(jobs)}\n\n" + "\n\n".join(answers_text) + "\n"
    )
    (EXPORT_DIR / text_name).write_text(text_attachment, encoding="utf-8")
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
        "path": str((EXPORT_DIR / text_name).relative_to(ROOT)),
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
def _batch_progress(worker: BatchWorker, keys: list[str]) -> None:
    """Live progress of a running batch, refreshed every two seconds without rerunning the page."""
    if not worker.is_alive():
        st.rerun()  # the whole page: export the result and show it
    # Read the checkpoint from disk: the worker's own copy may change while it is being read.
    saved = Checkpoint.load(worker.checkpoint.path) or worker.checkpoint
    scoring = worker.phase == "scoring"
    verb = "scored" if scoring else "generated"
    total = len(keys)
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
            st.button("Stopping after the block in progress…", disabled=True, icon=":material/hourglass_top:")
        elif st.button("Stop", icon=":material/stop_circle:",
                       help="Finishes the block in progress, then stops. Every block written so far is saved and "
                            "exported; press Resume to continue."):
            worker.stop()
            st.rerun(scope="fragment")


def _batch_status(plan: Plan, jobs: list[dict], path: Path) -> tuple[Checkpoint | None, BatchWorker | None]:
    """Export a batch that has just ended and say how it ended, show a running one's progress, or offer
    a saved one's progress to resume."""
    keys = [job["key"] for job in jobs]
    checkpoint = Checkpoint.load(path)
    worker = running_batches().get(str(path))
    if worker and not worker.is_alive():
        running_batches().pop(str(path), None)
        checkpoint = Checkpoint.load(path) or worker.checkpoint
        saved = len(checkpoint.generated(keys))
        if worker.phase == "done":
            _export_batch(plan, jobs, checkpoint, partial=False)
            checkpoint.discard()
            checkpoint = None
            st.session_state[NOTICE] = ("success", f"All {len(keys)} blocks are done and exported below.")
        else:
            _export_batch(plan, jobs, checkpoint, partial=True)
            if worker.phase == "stopped":
                st.session_state[NOTICE] = ("warning", f"Stopped after {saved} of {len(keys)} blocks. They are "
                                                       "saved and exported below; press Resume to continue.")
            elif saved < len(keys):
                st.session_state[NOTICE] = ("error", f"Generation stopped at block {saved + 1} of {len(keys)}. "
                                                     f"The {saved} blocks written before it are saved and exported "
                                                     f"below. Fix the cause, then press Resume.\n\n{worker.error}")
            else:
                st.session_state[NOTICE] = ("error", f"Scoring stopped: {worker.error}. All {saved} blocks are "
                                                     "generated, saved and exported below. Press Resume to retry "
                                                     "scoring, or turn scoring off and Resume to finish without "
                                                     "scores.")
        worker = None
    if worker:
        _batch_progress(worker, keys)
    elif checkpoint:
        saved, scored = len(checkpoint.generated(keys)), len(checkpoint.scored(keys))
        with st.container(border=True):
            st.markdown(f":material/save: **Saved progress for this document:** {saved} of {len(keys)} blocks "
                        "generated" + (f", {scored} scored" if plan.score else "")
                        + (f". Resume continues from block {saved + 1}." if saved < len(keys)
                           else ". Resume finishes the batch."))
            if st.button("Discard and start over", icon=":material/delete:", type="tertiary"):
                checkpoint.discard()
                st.rerun()
    return checkpoint, worker


def _start_batch(plan: Plan, conn: Connection, jobs: list[dict], path: Path, checkpoint: Checkpoint | None,
                 system: str | None) -> None:
    keys = [job["key"] for job in jobs]
    if checkpoint is None:
        checkpoint = Checkpoint.start(
            path, {"source_file": plan.upload.name, "mode": plan.mode_tag, "blocks": len(jobs)}
        )
    by_key = {job["key"]: job for job in jobs}
    status = {"model": None, "note": None, "log": [], "released": False,
              "order": list(dict.fromkeys([conn.model, *conn.fallback_models]))}

    # These run on the worker's thread, so they must not call Streamlit; they report through `status`.
    def generate_block(key: str) -> tuple[str, str]:
        job = by_key[key]
        voice_samples = [job["previous"]]
        if job["kind"] == "answer_box":
            voice_samples.extend(slot.previous_answer for slot in plan.answer_slots if slot.previous_answer)
        else:
            voice_samples.extend(section.text for section in plan.blocks if section.text != job["previous"])
        voice_rule = person_instruction(*voice_samples)
        target_words = (
            count_words(job["previous"])
            if plan.match_existing_lengths and job["previous"].strip()
            else plan.answer_words
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
                f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
                f"Question to answer:\n{job['question']}\n\n"
                f"Previous answer in this right-hand box (may be blank):\n{job['previous'] or '(blank)'}\n\n"
                f"Other document context:\n{plan.context_text}"
            )
        else:
            generation_prompt = (
                f"Rewrite only the existing {'Markdown text' if job['kind'] == 'markdown_block' else 'text'} block "
                f"labelled {job['label']}. "
                f"Aim for about {target_words} words. "
                f"{voice_rule} "
                "Keep its meaning and role in the document, using the surrounding document as context. "
                "Return only the "
                "replacement for this block. Do not output headings, questions, table rows, or any other block. "
                "This replacement is AI-generated red-team evaluation data.\n\n"
                f"Additional instructions:\n{plan.prompt.strip() or '(none)'}\n\n"
                f"Existing text to regenerate:\n{job['previous']}\n\n"
                f"Full document context:\n{plan.context_text}"
            )
        answer, used_model, model_errors = generate_with_fallback(
            conn.base_url, conn.api_key, status["order"], generation_prompt, plan.temperature, system,
            conn.local_url, conn.local_model,
        )
        if used_model != status["model"]:
            status["log"].append(f"Writing with {used_model}" + (
                f" after {len(model_errors)} model(s) failed. Last error: {model_errors[-1][:300]}"
                if model_errors else "."
            ))
            status["model"] = used_model
        if used_model == conn.local_model and not status["released"]:
            models.release()  # the local model needs the memory a loaded detector holds
            status["released"] = True
        # Later blocks start from the model that worked; once that is the local model, the
        # hosted ones are spent and every later block goes straight to it.
        order = status["order"]
        status["order"] = order[order.index(used_model) :] if used_model in order else []
        return clean_essay(answer), used_model

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

    worker = BatchWorker(
        checkpoint, keys, generate_block,
        score=(lambda text: score_text(text, plan.profile)) if plan.score else None,
        before_scoring=before_scoring, finish=score_document if plan.score else None, status=status,
    )
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
    try:
        with st.spinner(f"Generating with {conn.model}…"):
            essay, model, model_errors = generate_with_fallback(
                conn.base_url, conn.api_key, [conn.model, *conn.fallback_models], user_prompt, plan.temperature,
                system, conn.local_url, conn.local_model,
            )
    except APIError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    essay = clean_essay(essay)

    score = None
    if plan.score:
        try:
            if conn.local_model and model == conn.local_model:
                unload_local_model(conn.local_url, conn.local_model)  # the detector needs the memory it holds
            with st.spinner(f"Scoring with the {models.PROFILE_LABELS[plan.profile]} detector…"):
                score = models.get_detector(plan.profile).analyze(essay, sentences=False)
        except Exception as exc:  # Surface model/runtime failures without losing generated text.
            st.warning(f"The sample was generated, but detector scoring failed: {exc}")

    prompt_hash = hashlib.sha256(test_material.encode()).hexdigest()[:16]
    mode_tag = "regenerate_section" if target_section else "prompt"
    attack = f"{mode_tag}+{plan.style_name}"
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    export_name = f"AI_REDTEAM_{mode_tag}_{timestamp}_{prompt_hash}.txt"
    export_path = EXPORT_DIR / export_name
    attachment = (
        "Open Detector | AI-labelled red-team sample\n"
        f"Attack: {attack}\nModel: {model}\nSource: {upload.name if upload else 'prompt'}\n"
        f"Section: {target_section.label if target_section else 'new sample'}\n"
        f"Detector likelihood: {row['ai_probability'] if row['ai_probability'] is not None else 'not available'}\n\n"
        f"{essay}\n"
    )
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path.write_text(attachment, encoding="utf-8")
    document_attachment = document_filename = document_path = None
    if plan.export_document and upload and target_section:
        try:
            document_attachment = replace_document_section(upload.name, upload.data, target_section, essay)
            document_filename = f"AI_REDTEAM_{mode_tag}_{timestamp}_{prompt_hash}{upload.suffix}"
            document_export_path = EXPORT_DIR / document_filename
            document_export_path.write_bytes(document_attachment)
            document_path = str(document_export_path.relative_to(ROOT))
        except (ImportError, ValueError, IndexError, KeyError) as exc:
            st.warning(f"The answer was saved, but the document copy could not be exported: {exc}")
    st.session_state[RESULT] = {
        "essay": essay,
        "detect_text": essay,
        "row": row,
        "note": f"Written by {model} after {len(model_errors)} model(s) failed." if model_errors else None,
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


def _restore_last_result() -> None:
    """Show the most recent saved sample when the page opens in a new session."""
    if RESULT in st.session_state or not OUTPUT_PATH.is_file():
        return
    try:
        saved_rows = [json.loads(line) for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not saved_rows:
            return
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
            "path": str(saved_path.relative_to(ROOT)),
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
                column_config={"AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                max_value=1)},
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
        saved = [Path(result["path"]).name] + ([result["document_filename"]] if result.get("document_path") else [])
        st.caption(f"Saved as {' and '.join(saved)} in {Path(result['path']).parent}, and recorded as AI text in "
                   f"{OUTPUT_PATH.relative_to(ROOT)}.")


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
    if not OUTPUT_PATH.is_file():
        st.info("No samples yet. Generated samples are listed here with their scores.", icon=":material/history:")
        return
    rows = _saved_rows(str(OUTPUT_PATH), OUTPUT_PATH.stat().st_mtime)
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
                "Passed as human": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Mean AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
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
            "AI likelihood": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Text": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"All samples are in `{OUTPUT_PATH.relative_to(ROOT)}`, labelled as AI text.")


def render() -> None:
    load_env()
    profile = models.active_profile()
    conn = _connection()
    _restore_last_result()

    st.title("Red team", anchor=False)
    st.markdown("Generate labelled AI samples and see how the detector scores them. Everything generated here is "
                "saved as AI-labelled evaluation data.")
    st.caption(":material/shield: For detector evaluation only: don't submit generated work as your own.")

    generate_tab, saved_tab = st.tabs(["Generate", "Saved samples"])
    with generate_tab:
        setup, output = st.columns([1.1, 1], gap="large")
        with setup:
            plan = _setup(profile)
        jobs = _batch_jobs(plan) if plan.batch else []
        path = _checkpoint_path(plan) if plan.batch else None
        checkpoint = worker = None
        with output:
            if plan.batch:
                checkpoint, worker = _batch_status(plan, jobs, path)

        with setup:
            if checkpoint:
                saved = len(checkpoint.generated([job["key"] for job in jobs]))
                label, icon = f"Resume · {saved} of {len(jobs)} blocks saved", ":material/play_arrow:"
            elif plan.rewrite_all_blocks:
                label, icon = f"Rewrite {len(jobs)} blocks", ":material/auto_awesome:"
            elif plan.fill_answer_boxes:
                label, icon = f"Fill {len(jobs)} answer boxes", ":material/auto_awesome:"
            else:
                label, icon = "Generate sample", ":material/auto_awesome:"
            clicked = st.button(label, type="primary", icon=icon, disabled=worker is not None, width="stretch")
            problem = _problem(plan, conn) if clicked else None
            if problem:
                st.error(problem)
            system = None
            if clicked and not problem and plan.use_profile:
                try:
                    system = style_instructions(load_style_profile(STYLE_PATH))
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
            if notice := st.session_state.get(NOTICE):
                getattr(st, notice[0])(notice[1])
            if result := st.session_state.get(RESULT):
                _show_result(result, profile)
            elif not worker:
                st.info("Set up a sample on the left and generate it. It appears here with its detector score.",
                        icon=":material/auto_awesome:")

    with saved_tab:
        _history()
