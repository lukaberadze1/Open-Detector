"""What a Word or PowerPoint file says about itself, apart from its words: its properties, editing
history, formatting and hidden text. Standard library only, so every build has it.

None of this feeds the score. `flag` findings are text a reader can't see, which has no innocent use
in most documents; `note` findings have ordinary explanations too (Save As resets the editing time, and
Google Docs exports have no Word history), so they say where to look and prove nothing."""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from xml.etree import ElementTree

from .normalize import normalize

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
CP = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"
DC = "{http://purl.org/dc/elements/1.1/}"
DCTERMS = "{http://purl.org/dc/terms/}"
EXTENDED = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"

MAX_PART_BYTES = 64 * 1024 * 1024  # uploads are untrusted: a small zip can unpack to gigabytes
TINY_HALF_POINTS = 4  # Word stores sizes in half points, so 2 pt
MIN_SHIFT_WORDS = 3  # a shorter run in another font is usually a name or a code term
MIN_WORDS_FOR_TIMING = 300
FAST_WORDS_PER_MINUTE = 100  # well past anyone's sustained typing and revising
MIN_PARAGRAPHS_FOR_SESSIONS = 5
SCRIPT_AUTHORS = {"python-docx", "python-pptx"}
EXAMPLES = 3


@dataclass(frozen=True)
class Finding:
    level: str  # "flag" or "note"
    title: str
    detail: str


@dataclass
class DocumentForensics:
    properties: dict[str, str | int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_document(filename: str, data: bytes) -> DocumentForensics | None:
    """Inspect a .docx or .pptx. None for any other type, or a file that can't be read."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ("docx", "pptx"):
        return None
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            return _docx(archive) if suffix == "docx" else _pptx(archive)
    except (zipfile.BadZipFile, ElementTree.ParseError, ValueError, KeyError, OSError):
        return None


def _part(archive: zipfile.ZipFile, name: str) -> ElementTree.Element | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_PART_BYTES:
        raise ValueError(f"{name} is too large")
    raw = archive.read(info)
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:  # Office never writes these; entity bombs need them
        raise ValueError(f"{name} declares a document type")
    return ElementTree.fromstring(raw)


def _text(root: ElementTree.Element | None, tag: str) -> str:
    found = root.find(tag) if root is not None else None
    return (found.text or "").strip() if found is not None else ""


def _date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _properties(archive: zipfile.ZipFile, words: int) -> DocumentForensics:
    """The file's properties, and the notes that come from them alone."""
    core, app = _part(archive, "docProps/core.xml"), _part(archive, "docProps/app.xml")
    author, last = _text(core, f"{DC}creator"), _text(core, f"{CP}lastModifiedBy")
    created, modified = _text(core, f"{DCTERMS}created"), _text(core, f"{DCTERMS}modified")
    application = _text(app, f"{EXTENDED}Application")
    minutes = _text(app, f"{EXTENDED}TotalTime")
    values = {"Author": author, "Last saved by": last, "Created": created, "Modified": modified,
              "Application": application, "Revision": _text(core, f"{CP}revision"), "Editing minutes": minutes}
    result = DocumentForensics({key: value for key, value in values.items() if value})
    result.properties["Words"] = words
    note = lambda title, detail: result.findings.append(Finding("note", title, detail))  # noqa: E731

    if author.lower() in SCRIPT_AUTHORS or last.lower() in SCRIPT_AUTHORS:
        note("Made by a script", f"The file's author field is the library {author or last}, not a person.")
    if author and last and author.casefold() != last.casefold():
        note("Two names on the file", f"Created by {author}, last saved by {last}. Shared drafts and templates "
                                      "do this too.")
    start, end = _date(created), _date(modified)
    if start and end and start.tzinfo == end.tzinfo and end < start:
        note("Dates out of order", f"Last modified {modified}, before it was created ({created}).")
    if minutes.isdigit() and words >= MIN_WORDS_FOR_TIMING and int(minutes) * FAST_WORDS_PER_MINUTE < words:
        note("Little editing time", f"{words:,} words with {_count(int(minutes), 'minute')} of recorded editing. "
                                    "Text pasted in looks like this, and so does a copy made with Save As.")
    return result


def _hidden_characters(result: DocumentForensics, text: str) -> None:
    report = normalize(text)[1]
    found = [_count(n, word) for n, word in ((report.invisible_chars, "invisible character"),
                                             (report.homoglyphs, "look-alike letter")) if n]
    if found:
        result.findings.append(Finding("flag", "Hidden characters",
                                       f"{' and '.join(found)} in the file's text. Typing doesn't produce these."))


def _words(text: str) -> int:
    return len(text.split())


def _run_text(run: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in run.iter(f"{W}t"))


def _has_fill(element: ElementTree.Element | None) -> bool:
    shading = element.find(f"{W}shd") if element is not None else None
    return shading is not None and shading.get(f"{W}fill", "auto").lower() not in ("auto", "ffffff")


def _is_on(prop: ElementTree.Element | None) -> bool:
    return prop is not None and prop.get(f"{W}val", "true").lower() not in ("0", "false", "off")


def _docx(archive: zipfile.ZipFile) -> DocumentForensics:
    document = _part(archive, "word/document.xml")
    if document is None:
        raise ValueError("no document part")
    paragraphs = list(document.iter(f"{W}p"))
    in_shaded_cell = {id(p) for cell in document.iter(f"{W}tc") if _has_fill(cell.find(f"{W}tcPr"))
                      for p in cell.iter(f"{W}p")}
    texts = ["".join(_run_text(run) for run in p.iter(f"{W}r")) for p in paragraphs]
    result = _properties(archive, sum(_words(text) for text in texts))
    _hidden_characters(result, "\n".join(texts))

    hidden, white, tiny = 0, 0, 0
    shifts = []
    fonts: Counter[str] = Counter()
    sessions: set[str] = set()
    written = 0
    for paragraph, text in zip(paragraphs, texts, strict=True):
        if text.strip():
            written += 1
            if session := paragraph.get(f"{W}rsidR"):
                sessions.add(session)
        shaded = id(paragraph) in in_shaded_cell or _has_fill(paragraph.find(f"{W}pPr"))
        looks: Counter[tuple[str, str]] = Counter()
        for run in paragraph.iter(f"{W}r"):
            n = _words(_run_text(run))
            if not n:
                continue
            props = run.find(f"{W}rPr")
            get = props.find if props is not None else lambda _: None
            size, color, face = get(f"{W}sz"), get(f"{W}color"), get(f"{W}rFonts")
            size_value = size.get(f"{W}val", "") if size is not None else ""
            font = face.get(f"{W}ascii", "") if face is not None else ""
            if _is_on(get(f"{W}vanish")):
                hidden += n
            if (color is not None and color.get(f"{W}val", "").lower() == "ffffff"
                    and not shaded and not _has_fill(props)):
                white += n
            if size_value.isdigit() and int(size_value) <= TINY_HALF_POINTS:
                tiny += n
            if font:
                fonts[font] += n
            if get(f"{W}rStyle") is None:
                looks[(font, size_value)] += n
        if sum(count >= MIN_SHIFT_WORDS for count in looks.values()) >= 2:
            shifts.append(text.strip())

    flag = lambda title, detail: result.findings.append(Finding("flag", title, detail))  # noqa: E731
    if hidden:
        flag("Hidden text", f"{_count(hidden, 'word')} marked hidden in Word. The detector scored them anyway.")
    if white:
        flag("White text", f"{_count(white, 'word')} in white on an unshaded background, so a reader can't see them.")
    if tiny:
        flag("Microscopic text", f"{_count(tiny, 'word')} at 2 pt or smaller.")
    if shifts:
        samples = "; ".join(f"“{shift[:70]}…”" for shift in shifts[:EXAMPLES])
        result.findings.append(Finding(
            "note", "Font changes inside paragraphs",
            f"{_count(len(shifts), 'paragraph')} change font or size partway through, which pasting can leave "
            f"behind: {samples}"))
    if sessions:
        result.properties["Editing sessions"] = len(sessions)
        if len(sessions) == 1 and written >= MIN_PARAGRAPHS_FOR_SESSIONS:
            result.findings.append(Finding(
                "note", "Written in one session",
                f"Word records one editing session across all {written} paragraphs. A long piece usually builds "
                "up over several, though a retyped or copied file shows one too."))
    changes = sum(1 for tag in ("ins", "del") for _ in document.iter(f"{W}{tag}"))
    if changes:
        result.properties["Tracked changes"] = changes
    if fonts:
        result.properties["Fonts set on text"] = ", ".join(name for name, _ in fonts.most_common(5))
    return result


def _pptx(archive: zipfile.ZipFile) -> DocumentForensics:
    slides = sorted(name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
    texts = []
    for name in slides:
        slide = _part(archive, name)
        texts += [node.text or "" for node in slide.iter(f"{A}t")]
    result = _properties(archive, sum(_words(text) for text in texts))
    result.properties["Slides"] = len(slides)
    _hidden_characters(result, "\n".join(texts))
    return result
