"""Data quality gates for the ingestion pipeline.

A pipeline without gates fails quietly: it writes garbage and the failure only
shows up later as a bad answer. Each check below returns a reason string when it
rejects, and every rejection is counted on the run record.
"""

import re
from dataclasses import dataclass

from .chunk import Chunk
from .sources.ecfr import SourceDoc

_RESERVED = re.compile(r"\[reserved\]", re.IGNORECASE)
_MOSTLY_PUNCT = re.compile(r"^[\W\d_]+$")


@dataclass(frozen=True)
class Rejection:
    ref: str
    reason: str


def check_document(doc: SourceDoc, min_chars: int = 200) -> Rejection | None:
    if len(doc.text) < min_chars:
        return Rejection(doc.source_ref, f"document too short ({len(doc.text)} chars)")
    if _RESERVED.search(doc.title) or _RESERVED.search(doc.text[:200]):
        return Rejection(doc.source_ref, "reserved or placeholder section")
    if not doc.url.startswith("https://"):
        return Rejection(doc.source_ref, "insecure or missing source url")
    return None


def check_chunk(chunk: Chunk, ref: str, max_tokens: int) -> Rejection | None:
    if _MOSTLY_PUNCT.match(chunk.content):
        return Rejection(ref, f"chunk {chunk.ordinal} has no words")
    if chunk.token_estimate > max_tokens:
        return Rejection(
            ref, f"chunk {chunk.ordinal} over budget ({chunk.token_estimate} > {max_tokens})"
        )
    return None


def check_embedding(vec: list[float], dim: int, ref: str) -> Rejection | None:
    if len(vec) != dim:
        return Rejection(ref, f"embedding dim {len(vec)} != {dim}")
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return Rejection(ref, "zero embedding")
    return None
