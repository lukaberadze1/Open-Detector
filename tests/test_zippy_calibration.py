"""Exploratory ZipPy baseline on the cached, labeled MAGE PubMed test subset."""

import random
from pathlib import Path

import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from opendetector.zippy import compression_readings


def _mage_test_arrow() -> Path | None:
    try:
        import os

        root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        matches = sorted(root.glob("datasets/yaful___mage/default/*/*/mage-test.arrow"))
        return matches[0] if matches else None
    except OSError:
        return None


@pytest.fixture(scope="module")
def pubmed_calibration_rows():
    arrow_path = _mage_test_arrow()
    if arrow_path is None:
        pytest.skip("Cached MAGE test split is not available locally; calibration requires no network.")
    ipc = pytest.importorskip("pyarrow.ipc")
    with arrow_path.open("rb") as source_file:
        table = ipc.open_stream(source_file).read_all()
    sources = table.column("src").to_pylist()
    texts = table.column("text").to_pylist()
    rows = []
    for source, label in (("pubmed_human", 0), ("pubmed_gpt4", 1)):
        indices = [index for index, item in enumerate(sources) if item == source]
        random.Random(7).shuffle(indices)
        rows.extend((label, texts[index]) for index in indices[:50])
    if len(rows) != 100:
        pytest.skip("Cached MAGE split does not include 50 examples from each PubMed source.")
    return rows


@pytest.mark.parametrize("feature", ["zippy_lzma", "zippy_zlib"])
def test_mage_pubmed_scores_have_a_reportable_holdout_baseline(pubmed_calibration_rows, feature):
    """Choose a threshold on half the rows and report performance on the other half.

    This is a calibration baseline, not a claim that ZipPy alone is a production detector.
    MAGE's source tags define human abstracts and GPT-4 generations; no labels are inferred
    from the compression readings themselves.
    """
    scored = [(label, compression_readings(text)[feature]) for label, text in pubmed_calibration_rows]
    assert all(value == value for _, value in scored)  # all sampled abstracts yield finite scores
    train = scored[:25] + scored[50:75]
    held_out = scored[25:50] + scored[75:]
    train_labels = [label for label, _ in train]
    threshold = max(
        sorted({value for _, value in train}),
        key=lambda candidate: (
            balanced_accuracy_score(train_labels, [int(value >= candidate) for _, value in train]),
            -candidate,  # choose the lower threshold consistently when training scores tie
        ),
    )
    labels = [label for label, _ in held_out]
    values = [value for _, value in held_out]
    auc = roc_auc_score(labels, values)
    balanced_accuracy = balanced_accuracy_score(labels, [int(value >= threshold) for value in values])
    print(f"{feature}: threshold={threshold:.6f}, holdout AUROC={auc:.3f}, "
          f"balanced accuracy={balanced_accuracy:.3f} (n=50)")
    # A modest floor catches a direction reversal or collapse while keeping this small sample exploratory.
    assert auc >= 0.60
    assert balanced_accuracy >= 0.55
