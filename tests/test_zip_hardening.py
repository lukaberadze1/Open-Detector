"""Adversarial ZIP cases for the generator context ingestion boundary."""

import io
import warnings
import zipfile

import pytest

from opendetector.generator_inputs import extract_generation_source


def make_zip(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def test_duplicate_filenames_keep_each_entry_without_colliding():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bundle = make_zip([
            ("document.txt", b"First entry."),
            ("document.txt", b"Second duplicate entry."),
        ])

    sections = extract_generation_source("bundle.zip", bundle)

    assert [section.text for section in sections] == ["First entry.", "Second duplicate entry."]


def test_unicode_filename_is_read_and_non_utf8_payload_is_rejected_cleanly():
    bundle = make_zip([("münchen_data.txt", "Grüß Gott".encode("latin-1"))])

    with pytest.raises(ValueError, match="text file must use UTF-8 encoding"):
        extract_generation_source("bundle.zip", bundle)

    valid = make_zip([("münchen_data.txt", "Grüß Gott".encode())])
    sections = extract_generation_source("bundle.zip", valid)
    assert sections[0].text == "Grüß Gott"
    assert "münchen_data.txt" in sections[0].label


def test_malformed_zip_is_rejected_as_a_value_error():
    with pytest.raises(ValueError, match="not a valid ZIP archive"):
        extract_generation_source("broken.zip", b"PK\x03\x04" + b"\x00" * 50)


def test_deeply_nested_zip_members_are_not_recursively_expanded():
    nested = make_zip([("notes.txt", b"deeply nested text")])
    for depth in range(20):
        nested = make_zip([(f"level-{depth}.zip", nested)])

    with pytest.raises(ValueError, match="no supported prose or source-code comments"):
        extract_generation_source("outer.zip", nested)
