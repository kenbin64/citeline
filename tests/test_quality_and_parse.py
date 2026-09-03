"""Quality gate and eCFR parser tests."""

from citeline.ingest.chunk import Chunk
from citeline.ingest.quality import check_chunk, check_document, check_embedding
from citeline.ingest.sources.ecfr import SourceDoc, parse_sections

SAMPLE_XML = b"""<DIV5 N="141" TYPE="PART">
  <DIV8 N="&#167; 141.62" TYPE="SECTION">
    <HEAD>&#167; 141.62 Maximum contaminant levels for inorganic contaminants.</HEAD>
    <P>(a) [Reserved]</P>
    <P>(b) The maximum contaminant levels are as follows: Arsenic 0.010.</P>
  </DIV8>
  <DIV8 N="&#167; 141.63" TYPE="SECTION">
    <HEAD>&#167; 141.63 [Reserved]</HEAD>
    <P>[Reserved]</P>
  </DIV8>
  <DIV8 N="&#167; 141.64" TYPE="APPENDIX">
    <HEAD>Not a section</HEAD>
    <P>Should be ignored because the type is not SECTION.</P>
  </DIV8>
</DIV5>"""


def test_parses_only_section_divisions() -> None:
    docs = parse_sections(SAMPLE_XML, title=40, part=141, version="2025-01-01")
    refs = [d.source_ref for d in docs]
    assert "40 CFR 141.62" in refs
    assert "40 CFR 141.64" not in refs  # wrong TYPE


def test_parsed_document_carries_its_own_subject() -> None:
    docs = parse_sections(SAMPLE_XML, title=40, part=141, version="2025-01-01")
    doc = next(d for d in docs if d.source_ref == "40 CFR 141.62")
    # The chunk must be self describing, or a retrieved fragment is unattributable.
    assert doc.text.startswith("40 CFR 141.62")
    assert "inorganic" in doc.title.lower()
    assert "0.010" in doc.text
    assert doc.url.startswith("https://www.ecfr.gov/")


def test_content_sha_changes_with_content() -> None:
    a = SourceDoc("ecfr", "40 CFR 141.62", "t", "https://x.gov", "v", "body one")
    b = SourceDoc("ecfr", "40 CFR 141.62", "t", "https://x.gov", "v", "body two")
    assert a.content_sha != b.content_sha


def test_document_gate_rejects_short_and_reserved() -> None:
    short = SourceDoc("ecfr", "40 CFR 141.1", "t", "https://x.gov", "v", "tiny")
    assert check_document(short) is not None

    reserved = SourceDoc(
        "ecfr", "40 CFR 141.63", "[Reserved]", "https://x.gov", "v", "x" * 400
    )
    assert check_document(reserved) is not None

    good = SourceDoc("ecfr", "40 CFR 141.62", "Arsenic", "https://x.gov", "v", "x" * 400)
    assert check_document(good) is None


def test_document_gate_rejects_insecure_url() -> None:
    doc = SourceDoc("ecfr", "40 CFR 141.62", "Arsenic", "http://x.gov", "v", "x" * 400)
    assert check_document(doc) is not None


def test_chunk_gate_rejects_wordless_and_oversized() -> None:
    assert check_chunk(Chunk(0, "--- (1) ...", 4), "ref", 400) is not None
    assert check_chunk(Chunk(0, "real words here", 900), "ref", 400) is not None
    assert check_chunk(Chunk(0, "real words here", 40), "ref", 400) is None


def test_embedding_gate_rejects_wrong_dim_and_zero_vector() -> None:
    assert check_embedding([0.1] * 384, 768, "ref") is not None
    assert check_embedding([0.0] * 768, 768, "ref") is not None
    assert check_embedding([0.1] * 768, 768, "ref") is None
