"""Document forensics on files built in memory."""

import io
import zipfile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from pptx import Presentation

from opendetector.forensics import MAX_PART_BYTES, inspect_document


def saved(document) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def titles(forensics, level=None) -> set[str]:
    return {f.title for f in forensics.findings if level in (None, f.level)}


def test_other_types_and_broken_files_give_none():
    assert inspect_document("a.md", b"# hi") is None
    assert inspect_document("a.docx", b"not a zip") is None


def test_properties_are_read():
    document = Document()
    document.core_properties.author = "Sam"
    document.core_properties.last_modified_by = "Alex"
    document.add_paragraph("A few words here.")
    result = inspect_document("a.docx", saved(document))
    assert result.properties["Author"] == "Sam"
    assert result.properties["Words"] == 4
    assert titles(result, "note") == {"Two names on the file"}


def test_script_author_is_noted():
    document = Document()
    document.core_properties.author = "python-docx"
    assert "Made by a script" in titles(inspect_document("a.docx", saved(document)))


def test_hidden_white_and_tiny_text_are_flagged():
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("hidden words here").font.hidden = True
    paragraph.add_run("white words").font.color.rgb = RGBColor(255, 255, 255)
    paragraph.add_run("tiny").font.size = Pt(1)
    assert titles(inspect_document("a.docx", saved(document)), "flag") == {
        "Hidden text", "White text", "Microscopic text"}


def test_white_text_on_a_shaded_cell_is_fine():
    document = Document()
    cell = document.add_table(rows=1, cols=1).cell(0, 0)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "000000")
    cell._tc.get_or_add_tcPr().append(shading)
    cell.paragraphs[0].add_run("a white heading").font.color.rgb = RGBColor(255, 255, 255)
    assert "White text" not in titles(inspect_document("a.docx", saved(document)))


def test_zero_width_characters_are_flagged():
    document = Document()
    document.add_paragraph("some​words with a tr​ick")
    assert "Hidden characters" in titles(inspect_document("a.docx", saved(document)), "flag")


def test_font_shift_needs_a_real_passage():
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("the writer's own words go here")
    paragraph.add_run("Name").font.name = "Courier"
    assert "Font changes inside paragraphs" not in titles(inspect_document("a.docx", saved(document)))
    paragraph.add_run("and then a pasted passage follows").font.name = "Arial"
    assert "Font changes inside paragraphs" in titles(inspect_document("a.docx", saved(document)))


def test_pptx_is_read():
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[0])
    slide.shapes.title.text = "tr​ick"
    result = inspect_document("a.pptx", saved(deck))
    assert result.properties["Slides"] == 1
    assert "Hidden characters" in titles(result)


def test_entity_declarations_and_oversized_parts_are_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", '<!DOCTYPE x [<!ENTITY a "b">]><x>&a;</x>')
    assert inspect_document("a.docx", buffer.getvalue()) is None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b" " * (MAX_PART_BYTES + 1))
    assert inspect_document("a.docx", buffer.getvalue()) is None
