"""Citations are formatted only from the caller's reference metadata."""

import re

import pytest

from opendetector.citations import citation_material, validate_references


def sample_references():
    return validate_references([
        {"author": "Smith, A.", "title": "A first paper", "year": 2020, "venue": "Journal One",
         "doi": "10.1000/example"},
        {"author": "Smith, A.", "title": "A second paper", "year": 2020, "venue": "Conference Two"},
    ])


def test_reference_numbers_and_bibtex_keys_are_unique_and_ordered():
    prompt, bibliography, bibtex = citation_material(sample_references())
    assert prompt.startswith("[1] Smith, A. (2020). A first paper.")
    assert "[2] Smith, A. (2020). A second paper." in prompt
    assert bibliography == "References:\n" + prompt
    keys = re.findall(r"@misc\{([^,]+),", bibtex)
    assert len(keys) == 2 and len(set(keys)) == 2
    assert "doi = {10.1000/example}" in bibtex


def test_generated_bibtex_parses_with_pybtex_when_installed():
    database = pytest.importorskip("pybtex.database")
    _, _, bibtex = citation_material(sample_references())
    parsed = database.parse_string(bibtex, "bibtex")
    assert len(parsed.entries) == 2
    assert [entry.fields["title"] for entry in parsed.entries.values()] == ["A first paper", "A second paper"]


@pytest.mark.parametrize("value", [
    {"author": "Not a list"},
    [{"author": "A", "title": "T", "year": 2020}],
    [{"author": "A", "title": "T", "year": "old", "venue": "V"}],
])
def test_invalid_reference_metadata_is_rejected(value):
    with pytest.raises(ValueError):
        validate_references(value)
