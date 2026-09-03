"""Chunker tests. These are the tests that would have caught the bug where a
regulation table gets cut in half and the retrieved chunk shows a contaminant
name with no number next to it."""

from citeline.ingest.chunk import Chunk, chunk_text, est_tokens


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("A short section about arsenic limits in water.", min_chars=10)
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0


def test_respects_paragraph_boundaries() -> None:
    paras = [f"Paragraph {i} " + "filler words here " * 20 for i in range(6)]
    chunks = chunk_text("\n\n".join(paras), target_tokens=100, overlap_tokens=0)
    assert len(chunks) > 1
    # No chunk may begin mid word, which is what fixed size slicing does.
    for c in chunks:
        assert not c.content.startswith(" ")
        assert c.content == c.content.strip()


def test_oversized_paragraph_splits_on_sentences() -> None:
    para = " ".join(f"Sentence number {i} states a requirement." for i in range(80))
    chunks = chunk_text(para, target_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        # Each piece should still end at a sentence boundary.
        assert c.content.rstrip().endswith(".")


def test_overlap_carries_context_forward() -> None:
    a = "First block. " + "alpha " * 60
    b = "Second block. " + "beta " * 60
    chunks = chunk_text(f"{a}\n\n{b}", target_tokens=50, overlap_tokens=20)
    assert len(chunks) >= 2
    # A later chunk should contain a tail of the previous one.
    assert any("alpha" in c.content and "beta" in c.content for c in chunks[1:])


def test_no_overlap_when_disabled() -> None:
    a = "AAA. " + "alpha " * 60
    b = "BBB. " + "beta " * 60
    chunks = chunk_text(f"{a}\n\n{b}", target_tokens=50, overlap_tokens=0)
    assert not any("alpha" in c.content and "beta" in c.content for c in chunks)


def test_ordinals_are_contiguous_after_filtering() -> None:
    text = "\n\n".join(["ok " * 60, "x", "ok " * 60, "y"])
    chunks = chunk_text(text, target_tokens=40, overlap_tokens=0, min_chars=50)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_content_sha_is_stable_and_distinct() -> None:
    a = Chunk(0, "identical text", 3)
    b = Chunk(5, "identical text", 3)
    c = Chunk(0, "different text", 3)
    assert a.content_sha == b.content_sha
    assert a.content_sha != c.content_sha
    assert len(a.content_sha) == 64


def test_token_estimate_is_positive() -> None:
    assert est_tokens("") == 1
    assert est_tokens("a" * 400) == 100
