import math

import pytest

from opendetector import stylometry
from opendetector.segment import split_sentences


def test_even_rhythm_has_low_burstiness():
    even = " ".join(["The committee reviewed the proposal in great detail today."] * 8)
    varied = ("We waited. The committee spent most of the afternoon arguing about whether the budget "
              "line for the new library wing could survive another round of cuts. Nobody won. "
              "Then the chair adjourned. It was late, everyone was hungry, and the coffee had run out an "
              "hour before. We left.")
    assert stylometry.sentence_length_cv(even, split_sentences(even)) == pytest.approx(0.0)
    assert stylometry.sentence_length_cv(varied, split_sentences(varied)) > 0.5


def test_burstiness_skips_headings_and_needs_enough_sentences():
    short = "One two three. Four five six."
    assert math.isnan(stylometry.sentence_length_cv(short, split_sentences(short)))
    # A heading has no closing punctuation, so it doesn't count towards the rhythm.
    text = "Results and discussion\n" + " ".join(["The samples were dried and weighed twice."] * 4)
    assert stylometry.sentence_length_cv(text, split_sentences(text)) == pytest.approx(0.0)


def test_mattr_rewards_varied_vocabulary_and_is_length_robust():
    repetitive = "the cat sat on the mat and the cat sat on the mat " * 20
    varied = ("quick brown foxes jump over lazy dogs while seven wizards juggle bright copper kettles near "
              "silent harbours where old sailors mend torn nets beside crooked lanterns and gulls argue ") * 3
    assert stylometry.mattr(varied) > stylometry.mattr(repetitive)
    assert stylometry.mattr(repetitive) == pytest.approx(stylometry.mattr(repetitive * 5), abs=0.01)


def test_ai_phrases_are_found_with_offsets():
    text = ("In today's fast-paced world, it is important to note that we must delve into the rich "
            "tapestry of ideas. This plays a crucial role.")
    found = stylometry.ai_phrases(text)
    assert [p.text for p in found] == ["In today's fast-paced world", "it is important to note", "delve",
                                       "tapestry", "plays a crucial role"]
    assert all(text[p.start : p.end] == p.text for p in found)
    assert stylometry.phrase_counts(text + " Delve!")["delve"] == 2


def test_everyday_student_connectives_are_not_ai_phrases():
    text = ("In conclusion, furthermore, moreover and additionally are words students use. The landscape "
            "near the farm was flat. A key role. My foster family garden.")
    assert stylometry.ai_phrases(text) == []


def test_features_bundle():
    text = "We delve here. " * 3 + "Plain words only in this one. " * 3
    feats = stylometry.features(text, split_sentences(text))
    assert set(feats) == set(stylometry.FEATURES)
    assert feats["style_aiisms"] == pytest.approx(math.log1p(1000 * 3 / 27))
