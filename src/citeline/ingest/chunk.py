"""Structure aware chunking.

Fixed size chunking cuts regulations in half mid sentence and mid table, which
is the usual reason a RAG system retrieves something that looks relevant and
answers wrongly. This chunker splits on paragraph boundaries first and only
falls back to sentence boundaries when a single paragraph is over budget, so a
retrieved chunk is always a whole thought.

Token counts are estimated, not tokenised. The estimate is calibrated against
the embedding model's tokeniser (roughly 4 characters per token for English
regulatory prose) and is used only to size chunks, never to bill anything.
"""

import hashlib
import re
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d])")
CHARS_PER_TOKEN = 4


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    token_estimate: int

    @property
    def content_sha(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _split_oversized(para: str, budget_chars: int) -> list[str]:
    """Break a single over budget paragraph on sentence boundaries."""
    sentences = _SENT.split(para)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + 1 + len(s) > budget_chars:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf)
    # A single sentence longer than the budget is kept whole rather than cut.
    return out or [para]


def chunk_text(
    text: str,
    target_tokens: int = 350,
    overlap_tokens: int = 60,
    min_chars: int = 120,
) -> list[Chunk]:
    budget_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    paragraphs: list[str] = []
    for raw in text.split("\n\n"):
        p = raw.strip()
        if not p:
            continue
        if len(p) > budget_chars:
            paragraphs.extend(_split_oversized(p, budget_chars))
        else:
            paragraphs.append(p)

    # Pack paragraphs up to the budget.
    packed: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + 2 + len(p) > budget_chars:
            packed.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        packed.append(buf)

    # Carry a tail of the previous chunk into the next so a fact split across a
    # boundary is still retrievable from one side of it.
    chunks: list[Chunk] = []
    for i, body in enumerate(packed):
        content = body
        if i > 0 and overlap_chars > 0:
            tail = packed[i - 1][-overlap_chars:].lstrip()
            if tail:
                content = f"{tail}\n\n{body}"
        chunks.append(Chunk(ordinal=i, content=content, token_estimate=est_tokens(content)))

    # Drop trailing fragments that are too short to carry meaning, unless the
    # whole document is short, in which case keep the one chunk we have.
    kept = [c for c in chunks if len(c.content) >= min_chars]
    if not kept and chunks:
        kept = chunks[:1]
    return [
        Chunk(ordinal=i, content=c.content, token_estimate=c.token_estimate)
        for i, c in enumerate(kept)
    ]
