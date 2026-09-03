"""Postgres access. A single asyncpg pool, created once per process.

The pool is module level rather than passed around because both the API and the
CLI need it and there is exactly one database.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg

from .config import settings
from .obs.logging import get

log = get(__name__)

_pool: asyncpg.Pool | None = None

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


async def pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        cfg = settings()
        _pool = await asyncpg.create_pool(
            cfg.database_url,
            min_size=cfg.db_pool_min,
            max_size=cfg.db_pool_max,
            command_timeout=60,
        )
        log.info("db.pool_open", min=cfg.db_pool_min, max=cfg.db_pool_max)
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("db.pool_closed")


@asynccontextmanager
async def connection() -> AsyncIterator[asyncpg.Connection]:
    p = await pool()
    async with p.acquire() as conn:
        yield conn


async def migrate() -> list[str]:
    """Apply every .sql file in sql/ in filename order.

    The migrations are written to be idempotent (CREATE ... IF NOT EXISTS), which
    keeps this honest and re-runnable without a version table to drift out of sync.
    """
    applied: list[str] = []
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"no migrations found in {SQL_DIR}")
    async with connection() as conn:
        for f in files:
            await conn.execute(f.read_text(encoding="utf-8"))
            applied.append(f.name)
            log.info("db.migrated", file=f.name)
    return applied


async def corpus_stats() -> dict[str, int | str | None]:
    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT (SELECT count(*) FROM documents)                    AS documents,
                   (SELECT count(*) FROM chunks)                       AS chunks,
                   (SELECT count(*) FROM chunks WHERE embedding IS NULL) AS unembedded,
                   (SELECT max(version) FROM documents)                AS version,
                   (SELECT max(finished_at)::text FROM ingest_runs
                     WHERE status = 'succeeded')                       AS last_ingest
            """
        )
    return dict(row) if row else {}
