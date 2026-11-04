"""Regression checks for the offline deterministic rule pipeline."""

import json

import pytest

from opendetector.core.deterministic_engine import DeterministicEngine


def write_rules(tmp_path, ngrams, pdtb=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ngrams.json").write_text(json.dumps(ngrams), encoding="utf-8")
    (tmp_path / "pdtb_markers.json").write_text(json.dumps(pdtb or {"relations": {}}), encoding="utf-8")
    return tmp_path


def test_default_rules_load_and_run_deterministically():
    engine = DeterministicEngine()
    text = "in order to continue, however, keep the 2024 result."
    expected = "to continue, nevertheless, keep the 2024 result."
    assert engine.transform(text) == expected
    assert engine.transform(text) == expected


def test_longest_non_overlapping_ngram_match_wins(tmp_path):
    rules = {"rules": [
        {"source": "in order to", "replacement": "to", "category": "purpose", "domains": []},
        {"source": "order to", "replacement": "for", "category": "purpose", "domains": []},
    ]}
    engine = DeterministicEngine(write_rules(tmp_path, rules))
    assert engine.disrupt_ngram_sequences("in order to proceed") == "to proceed"


def test_math_citations_and_code_fences_restore_exactly():
    engine = DeterministicEngine()
    equation = r"\begin{equation}in order to + x_1 = 2024\end{equation}"
    citation = r"\cite[see][p. 4]{smith2024}"
    code = "```python\nin order to = 2024\n```"
    text = f"as a result {equation} {citation} {code}"
    transformed = engine.transform(text)
    assert transformed == f"consequently {equation} {citation} {code}"


def test_uuid_protection_is_repeatable_and_covers_inline_display_math_and_refs():
    engine = DeterministicEngine()
    text = r"$x=1$ $$y=2$$ \(z=3\) \[w=4\] \eqref{eq:5}"
    protected_a = engine.protect_ast_latex(text)
    protected_b = engine.protect_ast_latex(text)
    assert protected_a == protected_b
    assert engine.restore_ast_latex(*protected_a) == text


def test_protected_token_loss_or_duplication_is_rejected():
    engine = DeterministicEngine()
    protected, tokens = engine.protect_ast_latex(r"$x=2$ and \cite{work}")
    token = next(iter(tokens))
    with pytest.raises(ValueError, match="changed or duplicated"):
        engine.restore_ast_latex(protected.replace(token, ""), tokens)
    with pytest.raises(ValueError, match="changed or duplicated"):
        engine.restore_ast_latex(protected + token, tokens)


def test_fact_changing_rules_are_rejected_when_loaded(tmp_path):
    rules = {"rules": [{"source": "no more than 5", "replacement": "at least 6",
                        "category": "claim", "domains": []}]}
    with pytest.raises(ValueError, match="change a number or negation"):
        DeterministicEngine(write_rules(tmp_path, rules))


def test_capitalized_rule_phrases_are_rejected_to_protect_names(tmp_path):
    rules = {"rules": [{"source": "OpenAI research", "replacement": "other research",
                        "category": "names", "domains": []}]}
    with pytest.raises(ValueError, match="must be lowercase"):
        DeterministicEngine(write_rules(tmp_path, rules))


def test_pdtb_rotation_stays_within_relation_group(tmp_path):
    pdtb = {"relations": {"contrast": ["however", "nevertheless", "nonetheless"]}}
    engine = DeterministicEngine(write_rules(tmp_path, {"rules": []}, pdtb))
    assert engine.rotate_pdtb_markers("however; nevertheless; nonetheless") == \
        "nevertheless; nonetheless; however"
    assert engine.rotate_pdtb_markers("However, this stays capitalized.") == \
        "However, this stays capitalized."


def test_fact_tokens_and_protected_spans_survive_the_full_pass():
    engine = DeterministicEngine()
    original = r"in order to preserve 2024 and not change x, use $x_1=2$ and \cite{work2024}."
    result = engine.transform(original)
    assert result == r"to preserve 2024 and not change x, use $x_1=2$ and \cite{work2024}."


def test_red_team_page_counts_the_texts_its_rule_swaps_change():
    from ui.redteam import _rule_swaps

    swapped, changed = _rule_swaps(["in order to continue, keep the 2024 result.", "Plain words stay."])
    assert swapped == ["to continue, keep the 2024 result.", "Plain words stay."]
    assert changed == 1
