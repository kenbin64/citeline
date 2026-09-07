"""The HTTP service.

Public endpoints, since this is exposed on a portfolio site:
  POST /query      full pipeline, retrieval plus grounded generation
  POST /retrieve   retrieval only, no model call, fast
  GET  /healthz    dependency health, used by the deploy gate
  GET  /stats      corpus and live query statistics
  GET  /metrics    Prometheus exposition

The service is behind nginx and bound to loopback. Rate limiting lives here
rather than in nginx so the limit is part of the application contract and is
testable.
"""

import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import db
from ..config import settings
from ..generate import precomputed
from ..generate.answer import answer_question
from ..obs import logging as obslog
from ..obs.metrics import CORPUS_CHUNKS, CORPUS_DOCS, QUERIES, TOP_SCORE
from ..retrieve.hybrid import RetrievalResult, retrieve
from .schemas import (
    ConsideredOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveResponse,
    StatsResponse,
)

obslog.configure()
log = obslog.get(__name__)

# question -> deque of timestamps, per client address.
_hits: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await db.pool()
    stats = await db.corpus_stats()
    CORPUS_DOCS.set(float(stats.get("documents") or 0))
    CORPUS_CHUNKS.set(float(stats.get("chunks") or 0))
    log.info("api.start", **{k: str(v) for k, v in stats.items()})
    yield
    await db.close()


app = FastAPI(
    title="citeline",
    version="0.1.0",
    description=(
        "Retrieval augmented question answering over 40 CFR Part 141, the US "
        "National Primary Drinking Water Regulations. Answers carry citations "
        "or the service abstains."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _rate_limit(request: Request) -> None:
    cfg = settings()
    who = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    who = who.split(",")[0].strip()
    now = time.monotonic()
    q = _hits[who]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= cfg.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit is {cfg.rate_limit_per_minute} requests per minute",
        )
    q.append(now)


async def _log_query(
    question: str,
    answered: bool,
    abstained: bool,
    top_score: float,
    citations: list[str],
    retrieve_ms: int,
    generate_ms: int,
) -> None:
    async with db.connection() as conn:
        await conn.execute(
            """
            INSERT INTO query_log
                   (question, answered, abstained, top_score, citations,
                    retrieve_ms, generate_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            question,
            answered,
            abstained,
            top_score,
            citations,
            retrieve_ms,
            generate_ms,
        )


def _as_str(value: object) -> str | None:
    """corpus_stats() returns values straight from the database, where a date is
    a date and a version may be an int. The response model promises a string, so
    the conversion happens here rather than being asserted away."""
    return None if value is None else str(value)


def _considered(result: RetrievalResult) -> list[ConsideredOut]:
    """The retriever's candidates in the shape both endpoints return.

    Written once rather than twice so /query and /retrieve can never drift into
    describing the same passage differently, and typed rather than left as bare
    dicts so the two response models keep checking it.
    """
    return [
        ConsideredOut(
            ref=c.source_ref,
            title=c.title,
            score=round(c.score, 5),
            similarity=round(c.similarity, 4) if c.similarity is not None else None,
            found_by=c.found_by,
        )
        for c in result.candidates
    ]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    _rate_limit(request)
    cfg = settings()

    # A curated question we already have an answer for. Served instantly and
    # LABELLED as precomputed, because a stored answer presented as a live one
    # would be its own small dishonesty in a project about not bluffing.
    stored = precomputed.get(req.question)
    if stored is not None:
        QUERIES.labels(outcome="precomputed").inc()
        return QueryResponse(**{**stored, "precomputed": True})

    # Generation is disabled on the public host. Measured there: one grounded
    # answer over a realistic prompt ran past four minutes, because the CPU is
    # a virtual one with no AVX or SSE4.2 and llama.cpp falls back to scalar
    # arithmetic. Rather than hold a request open for minutes, answer with what
    # is genuinely fast: the live retrieval and the gate decision.
    if not cfg.serve_generation:
        r = await retrieve(req.question)
        passed = r.max_similarity >= cfg.min_similarity and (
            r.lexical_matched or not cfg.require_lexical_match
        )
        QUERIES.labels(outcome="abstained" if not passed else "retrieval_only").inc()
        # Say which gate actually blocked it. Reporting a similarity failure for
        # a lexical block produced a demonstrably false sentence in public: a
        # question whose best passage scored 0.667 was told that 0.667 was below
        # 0.62. Two gates, two reasons.
        if passed:
            reason = (
                "the corpus covers this question, but this host does not "
                "generate answers on demand; see /examples for questions "
                "answered through the full pipeline"
            )
        elif r.max_similarity < cfg.min_similarity:
            reason = (
                f"best passage similarity {r.max_similarity:.3f} is below the "
                f"{cfg.min_similarity} threshold, so the corpus does not appear "
                "to cover this question"
            )
        else:
            reason = (
                f"the closest passage scored {r.max_similarity:.3f}, above the "
                f"{cfg.min_similarity} similarity threshold, but no passage "
                "contains the question's own terms, so the match is by topic "
                "rather than by subject"
            )
        answer = (
            "I do not have a sourced answer to that. The indexed regulations do "
            "not contain a passage that answers it, so rather than guess, this "
            "returns nothing."
            if not passed
            else "Retrieval ran and found supporting passages, listed below. "
            "This host does not run the language model on demand, so no prose "
            "answer was generated for this question."
        )
        return QueryResponse(
            question=req.question,
            answer=answer,
            abstained=not passed,
            reason=reason,
            citations=[],
            considered=_considered(r),
            top_score=round(r.top_score, 5),
            max_similarity=round(r.max_similarity, 4),
            retrieve_ms=r.retrieve_ms,
            generate_ms=0,
            model="",
            precomputed=False,
        )

    result = await answer_question(req.question)
    TOP_SCORE.observe(result.top_score)
    QUERIES.labels(outcome="abstained" if result.abstained else "answered").inc()
    await _log_query(
        req.question,
        answered=not result.abstained,
        abstained=result.abstained,
        top_score=result.top_score,
        citations=[c.source_ref for c in result.citations],
        retrieve_ms=result.retrieve_ms,
        generate_ms=result.generate_ms,
    )
    return QueryResponse(**result.as_dict())


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_only(req: QueryRequest, request: Request) -> RetrieveResponse:
    _rate_limit(request)
    cfg = settings()
    result = await retrieve(req.question)

    passed = result.max_similarity >= cfg.min_similarity and (
        result.lexical_matched or not cfg.require_lexical_match
    )
    if passed:
        gate_reason = "passed: a passage is close enough and the terms appear in the corpus"
    elif result.max_similarity < cfg.min_similarity:
        gate_reason = (
            f"blocked: best similarity {result.max_similarity:.3f} is below "
            f"the {cfg.min_similarity} threshold"
        )
    else:
        gate_reason = "blocked: no passage contains the question's terms"

    return RetrieveResponse(
        question=req.question,
        candidates=_considered(result),
        top_score=round(result.top_score, 5),
        max_similarity=round(result.max_similarity, 4),
        retrieve_ms=result.retrieve_ms,
        vector_hits=result.vector_hits,
        lexical_hits=result.lexical_hits,
        passed_gate=passed,
        threshold=cfg.min_similarity,
        gate_reason=gate_reason,
    )


@app.get("/examples")
async def examples() -> dict:
    """The questions that have a precomputed answer.

    The demo page renders these as one click buttons. Any other question still
    works: retrieval runs live against the real index.
    """
    store = precomputed.load()
    return {
        "count": len(store),
        "questions": [item["question"] for item in store.values()],
        "note": (
            "These were answered offline through the same pipeline the service "
            "uses, because this host has no GPU. Retrieval is always live."
        ),
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    cfg = settings()
    database = False
    stats: dict = {}
    try:
        stats = await db.corpus_stats()
        database = True
    except Exception as exc:
        log.error("health.db_failed", error=str(exc))

    embed_ok = gen_ok = False
    try:
        async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=5.0) as c:
            tags = (await c.get("/api/tags")).json().get("models", [])
            names = {m.get("name", "").split(":")[0] for m in tags}
            embed_ok = cfg.embed_model.split(":")[0] in names
            gen_ok = cfg.generate_model.split(":")[0] in names
    except Exception as exc:
        log.error("health.ollama_failed", error=str(exc))

    healthy = database and embed_ok and int(stats.get("chunks") or 0) > 0
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=database,
        embedding_model=embed_ok,
        generation_model=gen_ok,
        documents=int(stats.get("documents") or 0),
        chunks=int(stats.get("chunks") or 0),
        corpus_version=_as_str(stats.get("version")),
        last_ingest=_as_str(stats.get("last_ingest")),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    corpus = await db.corpus_stats()
    async with db.connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT count(*) AS total,
                   coalesce(avg(CASE WHEN abstained THEN 1.0 ELSE 0.0 END), 0) AS abstain_rate,
                   percentile_disc(0.5) WITHIN GROUP (ORDER BY retrieve_ms) AS p50
              FROM query_log
            """
        )
    return StatsResponse(
        documents=int(corpus.get("documents") or 0),
        chunks=int(corpus.get("chunks") or 0),
        unembedded=int(corpus.get("unembedded") or 0),
        corpus_version=_as_str(corpus.get("version")),
        last_ingest=_as_str(corpus.get("last_ingest")),
        queries_total=int(row["total"] or 0),
        abstention_rate=round(float(row["abstain_rate"] or 0), 4),
        median_retrieve_ms=int(row["p50"]) if row["p50"] is not None else None,
    )


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
