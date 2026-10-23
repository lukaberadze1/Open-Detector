from opendetector.normalize import normalize


def test_strips_and_counts_invisible_characters():
    text, report = normalize("The​ cat‍ sat﻿.")
    assert text == "The cat sat."
    assert report.invisible_chars == 3
    assert report.suspicious


def test_maps_homoglyphs_inside_latin_words():
    # Cyrillic а and о hidden in English words.
    text, report = normalize("The cаt sat on the mаt and lооked around.")
    assert text == "The cat sat on the mat and looked around."
    assert report.homoglyphs == 4


def test_maps_all_lookalike_word_in_latin_text():
    # A lone Cyrillic "а" standing in for the English article.
    text, report = normalize("This is а sentence written in English with many words.")
    assert " a sentence" in text
    assert report.homoglyphs == 1


def test_leaves_genuine_cyrillic_text_alone():
    original = "Привет, как дела? Это обычный русский текст."
    text, report = normalize(original)
    assert text == original
    assert report.homoglyphs == 0
    assert not report.suspicious


def test_collapses_spaces_but_keeps_paragraphs():
    text, _ = normalize("One   two. Three.\n\n\n\n  Next paragraph.  ")
    assert text == "One two. Three.\n\nNext paragraph."
