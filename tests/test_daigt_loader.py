import pandas as pd

from eval.datasets import COLUMNS, load_daigt


def _essay(tag: str, words: int = 160) -> str:
    return " ".join([tag] * words)


def _write(tmp_path, rows):
    path = tmp_path / "daigt.csv"
    pd.DataFrame(rows, columns=["text", "label", "prompt_name", "source", "RDizzl3_seven"]).to_csv(path, index=False)
    return str(path)


def test_columns_labels_and_model(tmp_path):
    path = _write(tmp_path, [(_essay("human"), 0, "Cars", "persuade_corpus", True),
                             (_essay("robot"), 1, "Cars", "llama2_chat", True)])
    df = load_daigt(path)
    assert list(df.columns) == COLUMNS
    assert dict(zip(df.label, df.model, strict=True)) == {0: "human", 1: "llama2_chat"}
    assert set(df.dataset) == {"daigt"} and set(df.domain) == {"Cars"}
    assert (df.id == df.group_key).all()


def test_drops_copies_differing_only_in_spacing_and_short_texts(tmp_path):
    text = _essay("same")
    path = _write(tmp_path, [(text, 0, "Cars", "train_essays", True),
                             (text.replace(" ", "  ", 1), 0, "Cars", "persuade_corpus", True),
                             ("too short", 0, "Cars", "persuade_corpus", True)])
    assert len(load_daigt(path)) == 1


def test_sample_balances_labels_and_spreads_ai_over_sources(tmp_path):
    rows = [(_essay(f"h{i}"), 0, "Cars", "persuade_corpus", True) for i in range(20)]
    rows += [(_essay(f"big{i}"), 1, "Cars", "big", True) for i in range(20)]
    rows += [(_essay(f"small{i}"), 1, "Cars", "small", True) for i in range(3)]
    df = load_daigt(_write(tmp_path, rows), n=12)
    assert df.label.value_counts().to_dict() == {0: 6, 1: 6}
    assert df[df.label == 1].model.value_counts().to_dict() == {"big": 3, "small": 3}


def test_ids_do_not_depend_on_the_sample(tmp_path):
    rows = [(_essay(f"h{i}"), 0, "Cars", "persuade_corpus", True) for i in range(6)]
    path = _write(tmp_path, rows)
    assert set(load_daigt(path, n=4, seed=1).id) <= set(load_daigt(path).id)
