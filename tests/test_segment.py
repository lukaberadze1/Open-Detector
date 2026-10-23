from opendetector.segment import Sentence, sentence_windows, split_sentences


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
    assert all(b.first <= a.last for a, b in zip(windows, windows[1:])), "consecutive windows should overlap"


def test_last_window_is_anchored_to_the_end():
    windows = sentence_windows(fake_sentences([40, 40, 40, 10]), target_words=100, stride_words=50)
    assert [(w.first, w.last) for w in windows] == [(0, 1), (1, 3)]
    assert windows[-1].words == 90  # filled backwards to full size, not a 50-word stub


def test_short_text_is_one_window_and_long_sentence_stands_alone():
    assert len(sentence_windows(fake_sentences([10, 10]), 100, 50)) == 1
    windows = sentence_windows(fake_sentences([10, 500, 10]), 100, 50)
    assert any(w.first == w.last == 1 for w in windows)
    assert set().union(*(range(w.first, w.last + 1) for w in windows)) == {0, 1, 2}
