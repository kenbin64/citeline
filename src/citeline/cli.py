"""Operator CLI. Everything the pipeline can do is reachable from here, which is
what makes it schedulable from a timer and debuggable by hand.

    citeline migrate
    citeline ingest --limit 20
    citeline ask "What is the MCL for arsenic?"
    citeline search "lead service line replacement"
    citeline stats
"""

import asyncio
import json

import typer

from . import db
from .config import settings
from .generate.answer import answer_question
from .ingest.pipeline import ingest_ecfr
from .obs import logging as obslog
from .retrieve.hybrid import retrieve

app = typer.Typer(add_completion=False, help="citeline operator commands")


def _run(coro: object) -> object:
    obslog.configure()
    return asyncio.run(coro)  # type: ignore[arg-type]


@app.command()
def migrate() -> None:
    """Apply the SQL schema. Idempotent."""

    async def go() -> None:
        applied = await db.migrate()
        typer.echo(json.dumps({"applied": applied}, indent=2))
        await db.close()

    _run(go())


@app.command()
def ingest(
    title: int = typer.Option(None, help="CFR title, defaults to config"),
    part: int = typer.Option(None, help="CFR part, defaults to config"),
    version: str = typer.Option(None, help="Corpus edition date, YYYY-MM-DD"),
    limit: int = typer.Option(None, help="Ingest only the first N sections"),
    concurrency: int = typer.Option(4, help="Concurrent embedding requests"),
) -> None:
    """Fetch, gate, chunk, embed and index the corpus. Safe to re-run."""

    async def go() -> None:
        stats = await ingest_ecfr(
            title=title, part=part, version=version, limit=limit, embed_concurrency=concurrency
        )
        typer.echo(json.dumps(stats.as_dict(), indent=2))
        await db.close()

    _run(go())


@app.command()
def ask(question: str) -> None:
    """Ask a question through the full pipeline."""

    async def go() -> None:
        result = await answer_question(question)
        typer.echo(json.dumps(result.as_dict(), indent=2))
        await db.close()

    _run(go())


@app.command()
def search(question: str, k: int = typer.Option(6, help="Results to return")) -> None:
    """Retrieval only. No model call, so this is the fast path."""

    async def go() -> None:
        result = await retrieve(question, final_k=k)
        typer.echo(
            json.dumps(
                {
                    "question": question,
                    "top_score": round(result.top_score, 5),
                    "max_similarity": round(result.max_similarity, 4),
                    "threshold": settings().min_similarity,
                    "lexical_hits": result.lexical_hits,
                    "retrieve_ms": result.retrieve_ms,
                    "candidates": [
                        {
                            "ref": c.source_ref,
                            "title": c.title,
                            "score": round(c.score, 5),
                            "similarity": round(c.similarity, 4)
                            if c.similarity is not None
                            else None,
                            "found_by": c.found_by,
                        }
                        for c in result.candidates
                    ],
                },
                indent=2,
            )
        )
        await db.close()

    _run(go())


@app.command()
def stats() -> None:
    """Corpus counts and the last successful ingest."""

    async def go() -> None:
        typer.echo(json.dumps(await db.corpus_stats(), indent=2, default=str))
        await db.close()

    _run(go())


@app.command()
def serve(
    host: str = typer.Option(None), port: int = typer.Option(None), reload: bool = False
) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    cfg = settings()
    uvicorn.run(
        "citeline.api.main:app",
        host=host or cfg.host,
        port=port or cfg.port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
