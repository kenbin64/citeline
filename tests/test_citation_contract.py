"""Tests for the citation contract.

This is the security boundary of the whole service. If verify_citations passes
something ungrounded, the product claim is false, so these cases are written as
attacks on it rather than as demonstrations that it works.
"""

from citeline.generate.answer import verify_citations

SUPPLIED = {1, 2, 3}


def test_accepts_a_properly_cited_answer() -> None:
    ok, reason, cited = verify_citations(
        "The maximum contaminant level for arsenic is 0.010 mg/L [1].", SUPPLIED
    )
    assert ok, reason
    assert cited == {1}


def test_accepts_multiple_citations_across_sentences() -> None:
    text = (
        "The lead action level is exceeded above 0.010 mg/L [1]. "
        "The copper action level is exceeded above 1.3 mg/L [2]."
    )
    ok, reason, cited = verify_citations(text, SUPPLIED)
    assert ok, reason
    assert cited == {1, 2}


def test_rejects_an_answer_with_no_citation() -> None:
    ok, reason, _ = verify_citations("The MCL for arsenic is 0.010 mg/L.", SUPPLIED)
    assert not ok
    assert "no citation" in reason


def test_rejects_an_invented_citation() -> None:
    """The model citing excerpt 9 when it was handed three is a fabricated source."""
    ok, reason, _ = verify_citations("Arsenic is limited to 0.010 mg/L [9].", SUPPLIED)
    assert not ok
    assert "not supplied" in reason


def test_rejects_a_smuggled_uncited_claim() -> None:
    """The realistic failure: one cited sentence, then an uncited assertion."""
    text = (
        "The MCL for arsenic is 0.010 mg/L [1]. "
        "Systems exceeding this must notify every customer within twenty four hours "
        "and replace all affected service lines within one year."
    )
    ok, reason, _ = verify_citations(text, SUPPLIED)
    assert not ok
    assert "no citation" in reason


def test_allows_short_connective_fragments() -> None:
    text = "Arsenic is capped at 0.010 mg/L [1]. That is the limit."
    ok, reason, _ = verify_citations(text, SUPPLIED)
    assert ok, reason


def test_citation_markers_do_not_inflate_sentence_length() -> None:
    """A fragment that is only brackets must not count as substantive."""
    text = "Arsenic is capped at 0.010 mg/L [1]. [1][2][3]"
    ok, reason, _ = verify_citations(text, SUPPLIED)
    assert ok, reason


def test_decimal_numbers_do_not_split_sentences() -> None:
    """Section numbers like 141.62 must not be read as a sentence boundary,
    or the checker would demand a citation for a fragment that has none."""
    text = "Under 40 CFR 141.62 the arsenic limit is 0.010 mg/L and applies to all systems [1]."
    ok, reason, _ = verify_citations(text, SUPPLIED)
    assert ok, reason


def test_empty_answer_is_rejected() -> None:
    ok, _, _ = verify_citations("", SUPPLIED)
    assert not ok
