from eval.overnight_summary import is_baseline, paired, passed, rejections, scoreboard, sign_test


def row(document, repeat, end, style="base", facts=80, start=0.8, model="m"):
    return {"document": document, "repeat": repeat, "style": style, "model": model, "start": start, "source": document,
            "requests": 2, "revise_requests": 5, "attack_requests": 0, "seconds": 120, "draft": f"draft {document}",
            "sentences": {"history": [start, end], "facts": facts, "text": f"{document} {repeat} {style}",
                          "outcomes": {"facts": 3, "scored": 1, "kept": 1}}}


def test_a_pass_needs_the_line_and_the_facts():
    assert passed(row("a", 1, 0.1))
    assert not passed(row("a", 1, 0.2))
    assert not passed(row("a", 1, 0.1, facts=40))


def test_variants_are_told_from_the_baseline():
    assert is_baseline("alexandria:finalized+writer+as-page+best3")
    assert not is_baseline("x+best3+drafts3")
    assert not is_baseline("x+best3+roundtrip-german")
    assert not is_baseline("x+best3+masked")


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert round(sign_test(8, 0), 4) == 0.0078
    assert sign_test(4, 4) == 1.0


def test_paired_uses_only_the_runs_both_reached():
    base = [row("a", 1, 0.5), row("a", 2, 0.9), row("b", 1, 0.5)]
    other = [row("a", 1, 0.3, "x+drafts3")]
    assert "-20 points" in paired(base, other) and "over 1 documents" in paired(base, other)
    assert paired(base, [row("c", 1, 0.3, "x+drafts3")]) == "no runs in common with the baseline"


def test_rejections_show_the_share_dropped_by_the_facts_check():
    assert "| a | 1 | 80 | 50 | 4 | 75% |" in "\n".join(rejections([row("a", 1, 0.5)]))


def test_setups_are_only_compared_within_a_model():
    rows = [row("a", 1, 0.5), row("a", 1, 0.2, "x+drafts3"), row("a", 1, 0.1, "x+drafts3", model="n")]
    text = scoreboard(rows, {"draft a": 0.9, "a 1 base": 0.8})
    assert "## n" in text and "No baseline runs" in text
    assert "90 > 80" in text and "| 7 | 2.0 |" in text
