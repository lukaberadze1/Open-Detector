import itertools

from opendetector.segment import Sentence, looks_like_prose, paragraph_blocks, sentence_windows, split_sentences


def texts(text):
    return [text[s.start : s.end] for s in split_sentences(text)]


def test_splits_on_terminal_punctuation():
    assert texts("It rained. We stayed in! Did you? Yes.") == ["It rained.", "We stayed in!", "Did you?", "Yes."]


def test_keeps_abbreviations_and_initials():
    text = "Dr. Smith met J. R. Tolkien in the U.S. last year. Then e.g. they talked."
    assert texts(text) == ["Dr. Smith met J. R. Tolkien in the U.S. last year.", "Then e.g. they talked."]


def test_newlines_are_boundaries_and_quotes_stay_attached():
    text = 'Title\n"Is it over?" she asked. It was.'
    assert texts(text) == ["Title", '"Is it over?" she asked.', "It was."]


def test_spans_and_word_counts():
    text = "  One two three.   Four five.  "
    sents = split_sentences(text)
    assert [(text[s.start : s.end], s.words) for s in sents] == [("One two three.", 3), ("Four five.", 2)]


def test_sentences_know_their_paragraph():
    text = "One here. Two here.\n\nThree here.\nFour here.\n \nFive."
    assert [s.paragraph for s in split_sentences(text)] == [0, 0, 1, 1, 2]


def test_windows_stay_inside_paragraphs_and_short_ones_merge():
    # paragraphs of 80, 20 (a heading-like line), 90 and 30 words
    sents = [Sentence(i * 10, i * 10 + 5, w, p) for i, (w, p) in
             enumerate([(40, 0), (40, 0), (20, 1), (45, 2), (45, 2), (30, 3)])]
    assert paragraph_blocks(sents, 60) == [(0, 1), (2, 5)]  # 20 merges forward; the 30-word tail merges back
    windows = sentence_windows(sents, target_words=100, stride_words=50, min_paragraph_words=60)
    assert all(w.last <= 1 or w.first >= 2 for w in windows), "no window spans the paragraph break"
    assert set().union(*(range(w.first, w.last + 1) for w in windows)) == set(range(6))
    assert paragraph_blocks(sents[:1], 60) == [(0, 0)], "a lone short paragraph is still one block"


def test_looks_like_prose():
    prose = ("The committee met on Tuesday to review the budget, and after a long discussion it agreed to "
             "postpone the vote until the next quarter. ") * 4
    wrapped = "\n".join(prose.split(" ")[i] + " " + " ".join(prose.split(" ")[i + 1 : i + 9])
                        for i in range(0, 80, 9)).replace("\n", "\n")  # hard-wrapped every ~9 words
    poem = "The frost is on the field\nthe crows are gone\nI walk the long road home\n" \
           "past the empty barn\nwhere my father stood\nand the light goes down\n"
    recipe = "2 cups flour\n1 cup sugar\n3 eggs\n1 tsp vanilla\n1 cup milk\nPinch of salt\n"
    assert looks_like_prose(prose)
    assert looks_like_prose(wrapped.lower()), "hard-wrapped prose is rejoined before measuring"
    assert not looks_like_prose(poem)
    assert not looks_like_prose(recipe)
    assert looks_like_prose("Too few lines\nto judge\n"), "short texts are left to the word minimum"


def fake_sentences(word_counts):
    out, pos = [], 0
    for n in word_counts:
        out.append(Sentence(pos, pos + 10, n))
        pos += 11
    return out


def test_windows_cover_every_sentence_and_overlap():
    sents = fake_sentences([20] * 20)  # 400 words
    windows = sentence_windows(sents, target_words=100, stride_words=50)
    covered = set()
    for w in windows:
        assert w.words <= 100
        covered.update(range(w.first, w.last + 1))
    assert covered == set(range(20))
    assert all(b.first <= a.last for a, b in itertools.pairwise(windows)), "consecutive windows should overlap"


def test_last_window_is_anchored_to_the_end():
    windows = sentence_windows(fake_sentences([40, 40, 40, 10]), target_words=100, stride_words=50)
    assert [(w.first, w.last) for w in windows] == [(0, 1), (1, 3)]
    assert windows[-1].words == 90  # filled backwards to full size, not a 50-word stub


def test_short_text_is_one_window_and_long_sentence_stands_alone():
    assert len(sentence_windows(fake_sentences([10, 10]), 100, 50)) == 1
    windows = sentence_windows(fake_sentences([10, 500, 10]), 100, 50)
    assert any(w.first == w.last == 1 for w in windows)
    assert set().union(*(range(w.first, w.last + 1) for w in windows)) == {0, 1, 2}
