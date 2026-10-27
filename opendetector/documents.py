"""Text extraction for user supplied Word and PowerPoint documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


@dataclass(frozen=True)
class DocumentSection:
    label: str
    text: str


def extract_document(filename: str, data: bytes) -> list[DocumentSection]:
    """Extract non-empty paragraphs or slides, keeping their source locations. Raises
    ValueError for an unsupported type or a file that can't be parsed as one."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    readers = {"docx": _docx_sections, "pptx": _pptx_sections}
    if suffix not in readers:
        raise ValueError("Upload a .docx Word document or .pptx PowerPoint presentation.")
    try:
        return readers[suffix](BytesIO(data))
    except ImportError:
        raise
    except Exception as exc:  # damaged or renamed files raise zip, XML and package errors
        raise ValueError(f"It isn't a valid .{suffix} file. It may be damaged, or another format "
                         f"renamed to .{suffix}.") from exc


def _docx_sections(stream: BytesIO) -> list[DocumentSection]:
    from docx import Document

    document = Document(stream)
    sections = [
        DocumentSection(f"Paragraph {i}", paragraph.text.strip())
        for i, paragraph in enumerate(document.paragraphs, 1)
        if paragraph.text.strip()
    ]
    for table_number, table in enumerate(document.tables, 1):
        for row_number, row in enumerate(table.rows, 1):
            value = _row_text(row.cells)
            if value:
                sections.append(DocumentSection(f"Table {table_number}, row {row_number}", value))
    return sections


def _pptx_sections(stream: BytesIO) -> list[DocumentSection]:
    from pptx import Presentation

    presentation = Presentation(stream)
    sections = []
    for slide_number, slide in enumerate(presentation.slides, 1):
        table_number = 0
        for shape_number, shape in enumerate(slide.shapes, 1):
            if getattr(shape, "has_text_frame", False):
                value = shape.text.strip()
                if value:
                    shape_name = getattr(shape, "name", "") or f"Text box {shape_number}"
                    sections.append(DocumentSection(f"Slide {slide_number}, {shape_name}", value))
            if getattr(shape, "has_table", False):
                table_number += 1
                for row_number, row in enumerate(shape.table.rows, 1):
                    value = _row_text(row.cells)
                    if value:
                        sections.append(DocumentSection(
                            f"Slide {slide_number}, Table {table_number}, row {row_number}", value
                        ))
    return sections


def _row_text(cells) -> str:
    """A table row as "a | b | c", or "" when every cell is blank."""
    values = [cell.text.strip() for cell in cells]
    return " | ".join(values) if any(values) else ""
