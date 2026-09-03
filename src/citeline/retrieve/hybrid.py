"""Hybrid retrieval: dense vectors, sparse keywords, fused by reciprocal rank.

Why hybrid rather than vector only. Regulatory text is full of exact tokens that
embeddings blur: numeric thresholds, contaminant names, section cross references,
units. Ask for "the MCL for arsenic" and a dense index happily returns three
sections about arsenic monitoring, because they are semantically close. The
lexical side pins the exact term, the dense side handles the paraphrase, and
fusion takes candidates that both methods like.

Fusion is Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009):

    score(d) = sum over retrievers of 1 / (k + rank(d))

RRF uses ranks, not scores, so it needs no score normalisation between two
retrievers whose scales have nothing to do with each other. That is exactly the
situation here: cosine distance and ts_rank_cd are not comparable numbers.
"""

import time
from dataclasses import dataclass

from .. import db
from ..config import settings
from ..ingest.embed import embed_query, to_pgvector
from ..obs.logging import get
from ..obs.metrics import RETRIEVE_SECONDS

log = get(__name__)


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    source_ref: str
    title: str
    url: str
    content: str
    score: float
    vector_rank: int | None
    lexical_rank: int | None
    # Raw cosine similarity from the dense retriever, 0..1. Kept separately from
    # the fused score because the two answer different questions: `score` says
    # how the candidate ranked, `similarity` says how close it actually is. Only
    # the second one can tell you the corpus does not contain the answer.
    similarity: float | None = None
    lexical_rank_score: float | None = None

    @property
    def found_by(self) -> str:
        if self.vector_rank is not None and self.lexical_rank is not None:
            return "both"
        if self.vector_rank is not None:
            return "vector"
        return "lexical"


async def vector_search(query_vec: list[float], k: int) -> list[dict]:
    """Dense retrieval over the HNSW index. `<=>` is pgvector cosine distance."""
    async with db.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.content, d.source_ref, d.title, d.url,
                   1 - (c.embedding <=> $1::vector) AS similarity
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.embedding IS NOT NULL
             ORDER BY c.embedding <=> $1::vector
             LIMIT $2
            """,
            to_pgvector(query_vec),
            k,
        )
    return [dict(r) for r in rows]


async def lexical_search(query: str, k: int) -> list[dict]:
    """Sparse retrieval over the GIN index.

    websearch_to_tsquery is used rather than plainto_tsquery because it tolerates
    real user input (quoted phrases, or/-) instead of erroring on punctuation.
    ts_rank_cd is the cover density variant, which rewards query terms appearing
    close together, and that is the right bias for a threshold in a table.
    """
    async with db.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.content, d.source_ref, d.title, d.url,
                   ts_rank_cd(c.tsv, websearch_to_tsquery('english', $1)) AS rank
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.tsv @@ websearch_to_tsquery('english', $1)
             ORDER BY rank DESC
             LIMIT $2
            """,
            query,
            k,
        )
    return [dict(r) for r in rows]


def fuse(
    vector_hits: list[dict],
    lexical_hits: list[dict],
    rrf_k: int,
    final_k: int,
) -> list[Candidate]:
    by_id: dict[int, dict] = {}
    scores: dict[int, float] = {}
    v_rank: dict[int, int] = {}
    l_rank: dict[int, int] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit["id"]
        by_id[cid] = hit
        v_rank[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for rank, hit in enumerate(lexical_hits, start=1):
        cid = hit["id"]
        by_id.setdefault(cid, hit)
        l_rank[cid] = rank
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    out: list[Candidate] = []
    seen_refs: set[str] = set()
    for cid, score in ordered:
        hit = by_id[cid]
        ref = hit["source_ref"]
        # One chunk per regulation section in the final set. Three chunks of the
        # same section crowd out the section that actually answers the question.
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        out.append(
            Candidate(
                chunk_id=cid,
                source_ref=ref,
                title=hit["title"],
                url=hit["url"],
                content=hit["content"],
                score=score,
                vector_rank=v_rank.get(cid),
                lexical_rank=l_rank.get(cid),
                similarity=float(hit["similarity"]) if hit.get("similarity") is not None else None,
                lexical_rank_score=float(hit["rank"]) if hit.get("rank") is not None else None,
            )
        )
        if len(out) >= final_k:
            break
    return out


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[Candidate]
    retrieve_ms: int
    vector_hits: int
    lexical_hits: int

    @property
    def top_score(self) -> float:
        """Best fused score. Use this for ranking, never for confidence."""
        return self.candidates[0].score if self.candidates else 0.0

    @property
    def max_similarity(self) -> float:
        """Highest raw cosine similarity among the candidates.

        This is the confidence signal. The fused score cannot serve as one: a
        dense retriever returns k rows for any question at all, so the top RRF
        score for a question about French geography looks much like the top RRF
        score for a question the corpus genuinely answers. Absolute similarity
        is what separates them.
        """
        sims = [c.similarity for c in self.candidates if c.similarity is not None]
        return max(sims) if sims else 0.0

    @property
    def lexical_matched(self) -> bool:
        """Whether the sparse retriever matched anything at all.

        A question using none of the corpus vocabulary produces zero full text
        matches, which is a second, independent signal that it is out of scope.
        """
        return self.lexical_hits > 0


async def retrieve(question: str, final_k: int | None = None) -> RetrievalResult:
    cfg = settings()
    started = time.perf_counter()

    query_vec = await embed_query(question)
    vector_hits = await vector_search(query_vec, cfg.vector_k)
    lexical_hits = await lexical_search(question, cfg.lexical_k)

    candidates = fuse(
        vector_hits,
        lexical_hits,
        rrf_k=cfg.rrf_k,
        final_k=final_k or cfg.final_k,
    )

    elapsed = time.perf_counter() - started
    RETRIEVE_SECONDS.observe(elapsed)
    log.info(
        "retrieve.done",
        question_chars=len(question),
        vector=len(vector_hits),
        lexical=len(lexical_hits),
        fused=len(candidates),
        ms=round(elapsed * 1000),
    )
    return RetrievalResult(
        candidates=candidates,
        retrieve_ms=round(elapsed * 1000),
        vector_hits=len(vector_hits),
        lexical_hits=len(lexical_hits),
    )
