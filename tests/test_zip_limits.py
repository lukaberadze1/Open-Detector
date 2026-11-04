"""ZIP generator inputs are bounded, read-only, and closed after parsing."""

import io
import zipfile
from types import SimpleNamespace

import pytest

from opendetector import generator_inputs
from opendetector.generator_inputs import extract_generation_source


def make_zip(entries: list[tuple[str, bytes]], compression=zipfile.ZIP_DEFLATED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def test_zip_text_is_read_and_binary_members_are_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = make_zip([
        ("notes/essay.txt", b"A paragraph for generation context."),
        ("assets/image.png", b"\x89PNG\r\n\x1a\n\x00\xff"),
    ])
    sections = extract_generation_source("bundle.zip", bundle)
    assert [section.text for section in sections] == ["A paragraph for generation context."]
    assert list(tmp_path.rglob("*")) == []  # no member is extracted to disk


def test_path_traversal_member_is_never_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = make_zip([("../../etc/passwd.txt", b"just text in the archive")])
    assert extract_generation_source("bundle.zip", bundle)[0].text == "just text in the archive"
    assert not (tmp_path.parent / "etc" / "passwd.txt").exists()


def test_compressed_oversized_member_is_rejected_before_reading():
    # A highly compressible member still exceeds the bounded expanded size.
    content = b"A" * (generator_inputs._MAX_ZIP_MEMBER + 1)
    bundle = make_zip([("large.txt", content)])
    with pytest.raises(ValueError, match="larger than 5 MB"):
        extract_generation_source("bundle.zip", bundle)


def test_total_expanded_size_and_member_count_are_bounded():
    total = generator_inputs._MAX_ZIP_TOTAL + 1
    bundle = make_zip([("large.txt", b"B" * total), ("small.txt", b"small")])
    with pytest.raises(ValueError, match="expands to more than 20 MB"):
        extract_generation_source("bundle.zip", bundle)

    bundle = make_zip([(f"{index}.txt", b"x") for index in range(generator_inputs._MAX_ZIP_MEMBERS + 1)])
    with pytest.raises(ValueError, match="more than 100 files"):
        extract_generation_source("bundle.zip", bundle)


def test_in_memory_stream_is_closed_after_parse(monkeypatch):
    real_bytes_io = io.BytesIO
    bundle = make_zip([("note.py", b"# context\n")])
    created = []

    def tracked_bytes_io(data):
        stream = real_bytes_io(data)
        created.append(stream)
        return stream

    monkeypatch.setattr(generator_inputs, "io", SimpleNamespace(BytesIO=tracked_bytes_io, StringIO=io.StringIO))
    extract_generation_source("bundle.zip", bundle)
    assert len(created) == 1
    assert created[0].closed
