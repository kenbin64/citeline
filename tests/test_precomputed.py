"""The precomputed answer store.

This exists because the deployment host has no GPU and a generated answer over
a realistic prompt was measured past four minutes. Answers for a curated set of
questions are generated once, offline, through the same pipeline, and served
instantly.

The risk it introduces is obvious and is exactly what this project is about: a
lookup that is too eager would return one question's answer for a different
question, which is a confident wrong answer with a citation attached. So the
match is exact after normalisation, and these tests try to make it misfire.
"""

import json

import pytest

from citeline.generate import precomputed


def test_normalise_ignores_case_spacing_and_punctuation() -> None:
    a = precomputed.normalise("What is the MCL for arsenic?")
    b = precomputed.normalise("  what is the mcl for arsenic  ")
    c = precomputed.normalise("What is the MCL for arsenic???")
    assert a == b == c


def test_normalise_does_not_collapse_different_questions() -> None:
    # Same shape, different contaminant. If normalisation ever made these equal
    # the store would answer one with the other.
    assert precomputed.normalise("What is the MCL for arsenic?") != precomputed.normalise(
        "What is the MCL for benzene?"
    )
    # Air versus drinking water is the near miss the demo set deliberately
    # includes; it must not normalise away.
    assert precomputed.normalise(
        "What is the national ambient air quality standard for benzene?"
    ) != precomputed.normalise("What is the drinking water standard for benzene?")


@pytest.fixture()
def store(tmp_path, monkeypatch) -> None:
    path = tmp_path / "precomputed.json"
    path.write_text(
        json.dumps(
            {
                "corpus_version": "2025-01-01",
                "answers": [
                    {
                        "question": "What is the MCL for arsenic?",
                        "answer": "0.010 mg/L [1]",
                        "abstained": False,
                        "reason": "",
                        "citations": [{"n": 1, "ref": "40 CFR 141.62", "title": "", "url": ""}],
                        "considered": [],
                        "top_score": 0.03,
                        "max_similarity": 0.85,
                        "retrieve_ms": 40,
                        "generate_ms": 90_000,
                        "model": "llama3.2:3b",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(precomputed, "store_path", lambda: path)
    precomputed.load(force=True)


def test_exact_question_hits(store: None) -> None:
    got = precomputed.get("What is the MCL for arsenic?")
    assert got is not None
    assert "0.010" in got["answer"]


def test_untidy_question_still_hits(store: None) -> None:
    assert precomputed.get("  what is the mcl for arsenic  ") is not None


def test_a_different_question_misses(store: None) -> None:
    # The important negative. A near neighbour must return nothing rather than
    # the arsenic answer, because a wrong cited answer is the failure this whole
    # project is built to avoid.
    assert precomputed.get("What is the MCL for benzene?") is None
    assert precomputed.get("What is the MCL for lead?") is None
    assert precomputed.get("arsenic") is None


def test_a_missing_store_is_not_an_error(tmp_path, monkeypatch) -> None:
    # The service must still start and serve retrieval with no store present.
    monkeypatch.setattr(precomputed, "store_path", lambda: tmp_path / "nope.json")
    assert precomputed.load(force=True) == {}
    assert precomputed.get("anything") is None


def test_a_corrupt_store_is_not_an_error(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "precomputed.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(precomputed, "store_path", lambda: bad)
    assert precomputed.load(force=True) == {}


def test_save_round_trips(tmp_path, monkeypatch) -> None:
    path = tmp_path / "out.json"
    monkeypatch.setattr(precomputed, "store_path", lambda: path)
    precomputed.save([{"question": "Q one?", "answer": "A", "abstained": False}], "2025-01-01")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["corpus_version"] == "2025-01-01"
    assert written["answers"][0]["question"] == "Q one?"
    # save() refreshes the cache, so the new answer is immediately reachable.
    assert precomputed.get("q one") is not None


def test_the_two_gate_reasons_are_distinguished() -> None:
    """A blocked question must be told WHY it was blocked, correctly.

    This went out in public saying "best passage similarity 0.667 is below the
    0.62 threshold", which is arithmetic nonsense. The question had been blocked
    by the lexical gate, not the similarity gate, and the message blamed the
    wrong one. On a service whose entire pitch is that it does not state things
    it cannot support, printing a false sentence is the worst possible bug.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "citeline" / "api" / "main.py"
    text = src.read_text(encoding="utf-8")

    at = text.index("if not cfg.serve_generation:")
    block = text[at : at + 2600]

    # Three outcomes, three distinct explanations.
    assert "elif r.max_similarity < cfg.min_similarity:" in block, (
        "the similarity reason must be guarded by an actual similarity comparison"
    )
    assert "no passage" in block and "contains the question's own terms" in block, (
        "a lexical block must say so rather than blaming similarity"
    )
    # The similarity sentence must not be reachable when similarity passed.
    similarity_sentence = "is below the"
    guard_pos = block.index("elif r.max_similarity < cfg.min_similarity:")
    assert block.index(similarity_sentence) > guard_pos, (
        "the 'is below the threshold' wording must sit inside the similarity branch"
    )
    assert re.search(r"if passed:", block), "the passing case is handled first"
