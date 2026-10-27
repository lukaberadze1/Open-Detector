import io
import zipfile

import pytest
from docx import Document
from pptx import Presentation
from pptx.util import Inches

from opendetector.documents import DocumentSection, extract_document


def docx_bytes():
    document = Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("   ")
    document.add_paragraph("Second paragraph.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "a", "b"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pptx_bytes():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # title only
    slide.shapes.title.text = "Title"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "x", "y"
    presentation.slides.add_slide(presentation.slide_layouts[6])  # blank, so skipped
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def test_docx_paragraphs_and_table_rows_keep_their_locations():
    assert extract_document("essay.DOCX", docx_bytes()) == [
        DocumentSection("Paragraph 1", "First paragraph."),
        DocumentSection("Paragraph 3", "Second paragraph."),
        DocumentSection("Table 1, row 1", "a | b"),
    ]


def test_pptx_shapes_and_table_rows_are_separate_sections():
    assert extract_document("deck.pptx", pptx_bytes()) == [
        DocumentSection("Slide 1, Title 1", "Title"),
        DocumentSection("Slide 1, Table 1, row 1", "x | y"),
    ]


@pytest.mark.parametrize("name", ["essay.txt", "no_extension"])
def test_unsupported_types_are_rejected(name):
    with pytest.raises(ValueError, match=r"Upload a \.docx"):
        extract_document(name, b"text")


def test_damaged_or_renamed_files_raise_value_error():
    other_zip = io.BytesIO()
    with zipfile.ZipFile(other_zip, "w") as z:
        z.writestr("notes.txt", "hi")
    for name, data in [("a.docx", b"not a zip"), ("a.pptx", other_zip.getvalue()), ("a.pptx", docx_bytes())]:
        with pytest.raises(ValueError, match=r"isn't a valid \.(docx|pptx) file"):
            extract_document(name, data)
