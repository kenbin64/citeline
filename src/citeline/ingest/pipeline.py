"""The ingestion pipeline: fetch -> parse -> quality gate -> chunk -> embed -> upsert.

Four properties make this a pipeline rather than a script, and each one is a
thing an interviewer will ask about:

1. Idempotent. Every document and chunk carries a SHA of its own content. A
   re-run over unchanged upstream text writes nothing and reports it as skipped,
   so the job is safe to schedule on a timer.
2. Incremental. Only documents whose SHA changed are re-chunked and re-embedded.
   Embedding is the expensive step, so this is the difference between a multi
   minute full build and a two second no-op.
3. Gated. Documents, chunks and embeddings each pass a quality check. Rejections
   are counted and logged, never silently written.
4. Observed. Each run opens a row in ingest_runs and closes it with a terminal
   status. A crashed run stays visible as running with no finished_at, which is
   what you want: a pipeline that fails quietly is worse than one that fails.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..config import settings
from ..obs.logging import get
from ..obs.metrics import CORPUS_CHUNKS, CORPUS_DOCS, INGEST_CHUNKS
from . import quality
from .chunk import chunk_text
from .embed import embed_documents, to_pgvector
from .sources.ecfr import SourceDoc, load

log = get(__name__)


@dataclass
class RunStats:
    run_id: int | None = None
    docs_seen: int = 0
    docs_written: int = 0
    docs_skipped: int = 0
    chunks_written: int = 0
    chunks_skipped: int = 0
    quality_failures: int = 0
    rejections: list[quality.Rejection] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "docs_seen": self.docs_seen,
            "docs_written": self.docs_written,
            "docs_skipped": self.docs_skipped,
            "chunks_written": self.chunks_written,
            "chunks_skipped": self.chunks_skipped,
            "quality_failures": self.quality_failures,
            "seconds": round(self.seconds, 2),
            "rejections": [{"ref": r.ref, "reason": r.reason} for r in self.rejections[:20]],
        }


async def _open_run(source: str, version: str) -> int:
    async with db.connection() as conn:
        run_id: int = await conn.fetchval(
            "INSERT INTO ingest_runs (source, version) VALUES ($1, $2) RETURNING id",
            source,
            version,
        )
        return run_id


async def _close_run(run_id: int, status: str, s: RunStats, error: str | None = None) -> None:
    async with db.connection() as conn:
        await conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = now(), status = $2, docs_seen = $3, docs_written = $4,
                   chunks_written = $5, chunks_skipped = $6, quality_failures = $7, error = $8
             WHERE id = $1
            """,
            run_id,
            status,
            s.docs_seen,
            s.docs_written,
            s.chunks_written,
            s.chunks_skipped,
            s.quality_failures,
            error,
        )


async def _existing_sha(conn: Any, doc: SourceDoc) -> tuple[int | None, str | None]:
    row = await conn.fetchrow(
        """SELECT id, content_sha FROM documents
            WHERE source = $1 AND source_ref = $2 AND version = $3""",
        doc.source,
        doc.source_ref,
        doc.version,
    )
    return (row["id"], row["content_sha"]) if row else (None, None)


async def _upsert_document(conn: Any, doc: SourceDoc) -> int:
    doc_id: int = await conn.fetchval(
        """
        INSERT INTO documents (source, source_ref, title, url, version, content_sha)
             VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (source, source_ref, version) DO UPDATE
             SET title = EXCLUDED.title, url = EXCLUDED.url,
                 content_sha = EXCLUDED.content_sha, fetched_at = now()
          RETURNING id
        """,
        doc.source,
        doc.source_ref,
        doc.title,
        doc.url,
        doc.version,
        doc.content_sha,
    )
    return doc_id


async def ingest_documents(docs: list[SourceDoc], embed_concurrency: int = 4) -> RunStats:
    """Ingest a batch of source documents. Safe to call repeatedly."""
    cfg = settings()
    s = RunStats()
    started = time.perf_counter()
    source = docs[0].source if docs else "unknown"
    version = docs[0].version if docs else cfg.ecfr_version
    s.run_id = await _open_run(source, version)

    try:
        for doc in docs:
            s.docs_seen += 1

            rej = quality.check_document(doc)
            if rej:
                s.quality_failures += 1
                s.rejections.append(rej)
                log.info("ingest.doc_rejected", ref=rej.ref, reason=rej.reason)
                continue

            async with db.connection() as conn:
                doc_id, prior_sha = await _existing_sha(conn, doc)

                if doc_id is not None and prior_sha == doc.content_sha:
                    s.docs_skipped += 1
                    n = await conn.fetchval(
                        "SELECT count(*) FROM chunks WHERE document_id = $1", doc_id
                    )
                    s.chunks_skipped += int(n or 0)
                    INGEST_CHUNKS.labels(result="skipped").inc(int(n or 0))
                    continue

            chunks = chunk_text(
                doc.text,
                target_tokens=cfg.chunk_target_tokens,
                overlap_tokens=cfg.chunk_overlap_tokens,
                min_chars=cfg.chunk_min_chars,
            )
            max_tok = cfg.chunk_target_tokens + cfg.chunk_overlap_tokens + 40
            good = []
            for c in chunks:
                crej = quality.check_chunk(c, doc.source_ref, max_tok)
                if crej:
                    s.quality_failures += 1
                    s.rejections.append(crej)
                    continue
                good.append(c)

            if not good:
                s.quality_failures += 1
                s.rejections.append(
                    quality.Rejection(doc.source_ref, "no usable chunks after gating")
                )
                continue

            # Embedding happens outside any held connection: it is slow, and
            # holding a pooled connection across it would starve the pool.
            vectors = await embed_documents(
                [c.content for c in good], concurrency=embed_concurrency
            )

            bad = False
            for _chunk, v in zip(good, vectors, strict=True):
                erej = quality.check_embedding(v, cfg.embed_dim, doc.source_ref)
                if erej:
                    s.quality_failures += 1
                    s.rejections.append(erej)
                    bad = True
            if bad:
                continue

            async with db.connection() as conn, conn.transaction():
                doc_id = await _upsert_document(conn, doc)
                # The document changed, so its existing chunks are stale by definition.
                await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
                for c, v in zip(good, vectors, strict=True):
                    await conn.execute(
                        """
                        INSERT INTO chunks
                               (document_id, ordinal, content, content_sha,
                                token_estimate, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6::vector)
                        ON CONFLICT (content_sha) DO NOTHING
                        """,
                        doc_id,
                        c.ordinal,
                        c.content,
                        c.content_sha,
                        c.token_estimate,
                        to_pgvector(v),
                    )
                s.docs_written += 1
                s.chunks_written += len(good)
                INGEST_CHUNKS.labels(result="written").inc(len(good))

            log.info(
                "ingest.doc_written",
                ref=doc.source_ref,
                chunks=len(good),
                progress=f"{s.docs_seen}/{len(docs)}",
            )

        s.seconds = time.perf_counter() - started
        await _close_run(s.run_id, "succeeded", s)

        stats = await db.corpus_stats()
        CORPUS_DOCS.set(float(stats.get("documents") or 0))
        CORPUS_CHUNKS.set(float(stats.get("chunks") or 0))
        log.info("ingest.done", **s.as_dict())
        return s

    except Exception as exc:
        s.seconds = time.perf_counter() - started
        if s.run_id is not None:
            await _close_run(s.run_id, "failed", s, error=f"{type(exc).__name__}: {exc}")
        log.error("ingest.failed", error=str(exc), **s.as_dict())
        raise


async def ingest_ecfr(
    title: int | None = None,
    part: int | None = None,
    version: str | None = None,
    limit: int | None = None,
    embed_concurrency: int = 4,
) -> RunStats:
    cfg = settings()
    title = title if title is not None else cfg.ecfr_title
    part = part if part is not None else cfg.ecfr_part
    version = version or cfg.ecfr_version

    log.info("ingest.fetch", title=title, part=part, version=version)
    docs = load(title, part, version)
    log.info("ingest.parsed", sections=len(docs))
    if limit:
        docs = docs[:limit]
    return await ingest_documents(docs, embed_concurrency=embed_concurrency)
