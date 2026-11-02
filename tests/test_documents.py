import io
import zipfile

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches

from opendetector.documents import (
    DocumentSection,
    extract_answer_slots,
    extract_document,
    replace_document_section,
    rewritable_blocks,
)


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


def test_docx_paragraphs_and_table_cells_keep_their_locations():
    assert extract_document("essay.DOCX", docx_bytes()) == [
        DocumentSection("Paragraph 1", "First paragraph.", "docx_paragraph", (0,)),
        DocumentSection("Paragraph 3", "Second paragraph.", "docx_paragraph", (2,)),
        DocumentSection("Table 1, row 1, cell 1", "a", "docx_table_cell", (0, 0, 0)),
        DocumentSection("Table 1, row 1, cell 2", "b", "docx_table_cell", (0, 0, 1)),
    ]


def test_pptx_shapes_and_table_cells_are_separate_sections():
    assert extract_document("deck.pptx", pptx_bytes()) == [
        DocumentSection("Slide 1, Title 1", "Title", "pptx_shape", (0, 0)),
        DocumentSection("Slide 1, Table 1, row 1, cell 1", "x", "pptx_table_cell", (0, 1, 0, 0)),
        DocumentSection("Slide 1, Table 1, row 1, cell 2", "y", "pptx_table_cell", (0, 1, 0, 1)),
    ]


def test_a_pptx_table_reads_as_question_and_answer_pairs():
    """python-pptx's row cells take no negative index, which once crashed every deck with a table."""
    (slot,) = extract_answer_slots("deck.pptx", pptx_bytes())
    assert (slot.question, slot.previous_answer, slot.target.locator) == ("x", "y", (0, 1, 0, 1))


@pytest.mark.parametrize("name", ["essay.rtf", "no_extension"])
def test_unsupported_types_are_rejected(name):
    with pytest.raises(ValueError, match=r"Upload a \.txt"):
        extract_document(name, b"text")


def test_damaged_or_renamed_files_raise_value_error():
    other_zip = io.BytesIO()
    with zipfile.ZipFile(other_zip, "w") as z:
        z.writestr("notes.txt", "hi")
    for name, data in [("a.docx", b"not a zip"), ("a.pptx", other_zip.getvalue()), ("a.pptx", docx_bytes())]:
        with pytest.raises(ValueError, match=r"isn't a valid \.(docx|pptx) file"):
            extract_document(name, data)


def worksheet_docx_bytes():
    document = Document()
    document.add_heading("Keeping bees", level=1)
    document.add_paragraph("Bees need a dry hive and a nearby source of water.")
    document.add_paragraph("What do bees eat?")
    document.add_paragraph("They live on nectar and pollen from flowers.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Why keep bees", "For the honey."
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_rewriting_a_word_document_skips_headings_questions_and_tables():
    assert [block.text for block in rewritable_blocks("w.docx", worksheet_docx_bytes())] == [
        "Bees need a dry hive and a nearby source of water.",
        "They live on nectar and pollen from flowers.",
    ]


def test_rewriting_slides_skips_titles_and_question_answer_box_pairs():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])  # title only
    slide.shapes.title.text = "Keeping bees"
    for left, top, width, text in ((0.5, 1.5, 9, "A hive needs shelter from the wind."),
                                   (0.5, 5.5, 4, "Describe the hive"), (5.5, 5.5, 4, "A wooden box.")):
        slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1)).text = text
    buffer = io.BytesIO()
    presentation.save(buffer)
    assert [block.text for block in rewritable_blocks("deck.pptx", buffer.getvalue())] == [
        "A hive needs shelter from the wind.",
    ]


def test_rewriting_markdown_keeps_headings_questions_and_tables():
    text = "# Bees\n\nBees make honey.\n\nQuestion: why?\n\n| a | b |\n|---|---|\n| c | d |\n\nThey sting.\n"
    assert [block.text for block in rewritable_blocks("notes.md", text.encode())] == [
        "Bees make honey.", "They sting.",
    ]


def test_plain_text_blocks_have_no_markup_and_a_paragraph_with_a_question_is_prose():
    text = ("# not a heading, just text\n\nWhy keep bees? Honey is one reason.\nThe garden is another.\n\n"
            "a | b is a pipe, not a table.\n\nWhat do bees eat?\n\nQuestion: and in winter\n")
    assert [block.text for block in rewritable_blocks("notes.txt", text.encode())] == [
        "# not a heading, just text",
        "Why keep bees? Honey is one reason.\nThe garden is another.",
        "a | b is a pipe, not a table.",
    ]
    assert extract_answer_slots("notes.txt", text.encode()) == []


def test_replacing_a_plain_text_block_keeps_the_rest_byte_for_byte():
    data = b"\xef\xbb\xbfFirst block.\r\n\r\nSecond block\r\nruns on.\r\n\r\nThird block."
    (_, second, _) = rewritable_blocks("notes.txt", data)
    assert second.text == "Second block\nruns on."
    assert replace_document_section("notes.txt", data, second, "New one.") == (
        b"\xef\xbb\xbfFirst block.\r\n\r\nNew one.\r\n\r\nThird block.")


def test_a_plain_text_file_must_be_utf8():
    with pytest.raises(ValueError, match="text file must use UTF-8"):
        extract_document("notes.txt", "caf\u00e9".encode("latin-1"))


def _bytes(document) -> bytes:
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _merged_title_docx() -> bytes:
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].merge(table.rows[0].cells[1]).text = "Section one"
    table.rows[1].cells[0].text = "What is it?"
    table.rows[1].cells[1].text = "An answer."
    return _bytes(document)


def test_a_merged_title_row_is_not_an_answer_box_and_reads_once():
    data = _merged_title_docx()
    assert [slot.question for slot in extract_answer_slots("a.docx", data)] == ["What is it?"]
    texts = [section.text for section in extract_document("a.docx", data)]
    assert texts == ["Section one", "What is it?", "An answer."]


def test_a_cell_merged_down_two_rows_is_one_answer_box():
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[1].cells[0].text = "First?", "Second?"
    table.rows[0].cells[1].merge(table.rows[1].cells[1]).text = "One answer."
    assert [slot.question for slot in extract_answer_slots("a.docx", _bytes(document))] == ["First?"]


def test_a_merged_pptx_title_row_is_not_an_answer_box():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(2)).table
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(0, 0).text = "Section one"
    table.cell(1, 0).text, table.cell(1, 1).text = "What is it?", "An answer."
    assert [slot.question for slot in extract_answer_slots("a.pptx", _bytes(presentation))] == ["What is it?"]


def test_a_pptx_shape_without_a_position_is_left_out_of_the_pairs():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(1), Inches(1))
    box.text = "Hello"
    box._element.spPr.remove(box._element.spPr.find(qn("a:xfrm")))
    assert extract_answer_slots("a.pptx", _bytes(presentation)) == []


@pytest.mark.parametrize("name", ["a.docx", "a.pptx"])
def test_answer_slots_of_a_damaged_file_raise_value_error(name):
    with pytest.raises(ValueError, match="valid"):
        extract_answer_slots(name, b"not a zip")


def test_replacing_a_word_paragraph_replaces_its_link_text_too():
    document = Document()
    paragraph = document.add_paragraph("Before the link ")
    link, run, text = OxmlElement("w:hyperlink"), OxmlElement("w:r"), OxmlElement("w:t")
    text.text = "LINK"
    run.append(text)
    link.append(run)
    link.set(qn("w:anchor"), "x")
    paragraph._p.append(link)
    data = _bytes(document)
    section = extract_document("a.docx", data)[0]
    assert section.text == "Before the link LINK"
    replaced = replace_document_section("a.docx", data, section, "New words.")
    assert extract_document("a.docx", replaced)[0].text == "New words."


def test_a_markdown_header_row_and_short_separator_are_not_answers():
    data = b"| Topic | Thoughts |\n|-|:-|\n| Cats | good |\n"
    assert [(slot.question, slot.previous_answer) for slot in extract_answer_slots("a.md", data)] == [("Cats", "good")]


def test_markdown_front_matter_and_code_are_not_prose():
    data = (b"---\ntitle: Notes\n---\nFirst real paragraph.\n\n```python\nx = a | b\n```\n\n"
            b"| Question | Answer |\n|---|---|\n| Why? | Because. |\n\n~~~\n| Fake? | row |\n~~~\nLast paragraph.\n")
    texts = [section.text for section in extract_document("a.md", data)]
    assert "First real paragraph." in texts and "Last paragraph." in texts
    assert not any("title:" in text or "x = a" in text or "Fake" in text for text in texts)
    assert [slot.question for slot in extract_answer_slots("a.md", data)] == ["Why?"]
    # Locators still point at the file's own lines.
    block = next(section for section in extract_document("a.md", data) if section.text == "Last paragraph.")
    replaced = replace_document_section("a.md", data, block, "Changed.")
    assert replaced.endswith(b"~~~\nChanged.\n") and b"title: Notes" in replaced


def test_a_leading_rule_is_not_front_matter():
    data = b"---\nA paragraph between rules.\n\n---\nAnother.\n"
    assert any("A paragraph between rules." in section.text for section in extract_document("a.md", data))
