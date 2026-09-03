"""Reciprocal rank fusion tests.

Fusion is the piece most likely to be quietly wrong: it will still return
plausible looking results if the maths is off, so the properties are asserted
directly rather than eyeballed.
"""

from citeline.retrieve.hybrid import fuse


def _hit(cid: int, ref: str) -> dict:
    return {
        "id": cid,
        "content": f"content {cid}",
        "source_ref": ref,
        "title": f"title {ref}",
        "url": f"https://example.gov/{ref}",
    }


def test_agreement_beats_a_single_strong_hit() -> None:
    """A document both retrievers rank second should beat one only vector ranks first.

    This is the entire reason for using fusion, so it gets a test.
    """
    vector = [_hit(1, "A"), _hit(2, "B")]
    lexical = [_hit(3, "C"), _hit(2, "B")]
    out = fuse(vector, lexical, rrf_k=60, final_k=3)
    assert out[0].source_ref == "B"
    assert out[0].found_by == "both"


def test_scores_are_descending() -> None:
    vector = [_hit(i, f"R{i}") for i in range(1, 6)]
    lexical = [_hit(i, f"R{i}") for i in range(5, 0, -1)]
    out = fuse(vector, lexical, rrf_k=60, final_k=5)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_score_matches_the_formula() -> None:
    out = fuse([_hit(1, "A")], [_hit(1, "A")], rrf_k=60, final_k=1)
    assert abs(out[0].score - (1 / 61 + 1 / 61)) < 1e-12


def test_deduplicates_by_source_section() -> None:
    """Three chunks of one section must not fill the whole context window."""
    vector = [_hit(1, "A"), _hit(2, "A"), _hit(3, "A"), _hit(4, "B")]
    out = fuse(vector, [], rrf_k=60, final_k=4)
    refs = [c.source_ref for c in out]
    assert refs == ["A", "B"]


def test_respects_final_k() -> None:
    vector = [_hit(i, f"R{i}") for i in range(1, 21)]
    out = fuse(vector, [], rrf_k=60, final_k=6)
    assert len(out) == 6


def test_empty_inputs_give_empty_output() -> None:
    assert fuse([], [], rrf_k=60, final_k=6) == []


def test_found_by_labels() -> None:
    out = fuse([_hit(1, "A")], [_hit(2, "B")], rrf_k=60, final_k=2)
    labels = {c.source_ref: c.found_by for c in out}
    assert labels == {"A": "vector", "B": "lexical"}
