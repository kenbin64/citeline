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
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import db
from ..config import settings
from ..generate.answer import answer_question
from ..obs import logging as obslog
from ..obs.metrics import CORPUS_CHUNKS, CORPUS_DOCS, QUERIES, TOP_SCORE
from ..retrieve.hybrid import retrieve
from .schemas import (
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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    _rate_limit(request)
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
        candidates=[
            {
                "ref": c.source_ref,
                "title": c.title,
                "score": round(c.score, 5),
                "similarity": round(c.similarity, 4) if c.similarity is not None else None,
                "found_by": c.found_by,
            }
            for c in result.candidates
        ],
        top_score=round(result.top_score, 5),
        max_similarity=round(result.max_similarity, 4),
        retrieve_ms=result.retrieve_ms,
        vector_hits=result.vector_hits,
        lexical_hits=result.lexical_hits,
        passed_gate=passed,
        threshold=cfg.min_similarity,
        gate_reason=gate_reason,
    )


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
        corpus_version=stats.get("version"),
        last_ingest=stats.get("last_ingest"),
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
        corpus_version=corpus.get("version"),
        last_ingest=corpus.get("last_ingest"),
        queries_total=int(row["total"] or 0),
        abstention_rate=round(float(row["abstain_rate"] or 0), 4),
        median_retrieve_ms=int(row["p50"]) if row["p50"] is not None else None,
    )


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
