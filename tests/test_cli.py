"""Command-line tests with a stand-in detector, so no model downloads are needed."""

import io
import json

import pytest
from docx import Document

from opendetector import Detector, DetectorConfig
from opendetector.__main__ import main
from tests.test_detector import AI, HUMAN, KeywordClassifier, LengthBinoculars


@pytest.fixture
def stub_detector(monkeypatch):
    configs = []

    def from_config(cls, config=None, load_combiner=True):
        configs.append(config)
        return Detector([KeywordClassifier()], LengthBinoculars(), None, DetectorConfig(min_words=40))

    monkeypatch.setattr(Detector, "from_config", classmethod(from_config))
    return configs


@pytest.fixture
def worksheet(tmp_path):
    document = Document()
    document.add_paragraph("Instructions. " + HUMAN * 6)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Explain your plan.", AI * 6
    table.cell(1, 0).text, table.cell(1, 1).text = "Name", "Sam"
    path = tmp_path / "worksheet.docx"
    document.save(path)
    return path


def run(capsys, *argv):
    main(list(argv))
    return json.loads(capsys.readouterr().out)


def test_document_gets_a_verdict_and_every_section_a_rating(stub_detector, worksheet, capsys):
    report = run(capsys, str(worksheet), "--no-sentences")
    assert report["ai_probability"] is not None and report["words"] > 150
    assert report["sections_extracted"] == 5
    assert [s["label"] for s in report["sections"]] == [
        "Paragraph 1 to Table 1, row 1, cell 1 (2 parts)", "Table 1, row 1, cell 2 to Table 1, row 2, cell 2 (3 parts)"
    ], "the short cells are rated with their neighbours, not left out as too short"
    assert sum(s["parts"] for s in report["sections"]) == 5
    assert stub_detector[0].glyph_model is not None  # full profile by default


def test_answers_only_leaves_out_the_template(stub_detector, worksheet, capsys):
    report = run(capsys, str(worksheet), "--answers-only", "--profile", "slim", "--device", "mps")
    assert report["sections_extracted"] == 2  # the long answer and "Sam"
    assert all("grandmother" not in s["text"] for s in report["sentences"])
    assert stub_detector[0].glyph_model is None and stub_detector[0].device == "mps"


def test_answers_only_rejects_plain_text_and_empty_worksheets(stub_detector, tmp_path):
    essay = tmp_path / "essay.txt"
    essay.write_text(HUMAN * 6)
    with pytest.raises(SystemExit):
        main([str(essay), "--answers-only"])
    blank = tmp_path / "blank.docx"
    Document().save(blank)
    with pytest.raises(SystemExit):
        main([str(blank), "--answers-only"])
    assert stub_detector == []


def test_plain_text_from_stdin_has_no_sections(stub_detector, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(HUMAN * 6))
    report = run(capsys)
    assert report["verdict"] == "likely_human" and "sections" not in report


def test_a_missing_or_binary_file_is_a_clear_error(tmp_path, capsys):
    binary = tmp_path / "a.pdf"
    binary.write_bytes(b"\xff\xfe\x00")
    for path, message in ((tmp_path / "missing.txt", "can't read"), (binary, "isn't UTF-8")):
        with pytest.raises(SystemExit) as stop:
            main([str(path)])
        assert stop.value.code == 2
        assert message in capsys.readouterr().err


def test_a_word_file_reports_its_forensics(stub_detector, tmp_path, capsys):
    document = Document()
    document.core_properties.author = "python-docx"
    document.add_paragraph(HUMAN)
    path = tmp_path / "essay.docx"
    document.save(path)
    main([str(path)])
    report = json.loads(capsys.readouterr().out)
    assert any(f["title"] == "Made by a script" for f in report["forensics"]["findings"])
