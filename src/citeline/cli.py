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
from pathlib import Path

import typer

from . import db
from .config import settings
from .generate.answer import answer_question
from .ingest.pipeline import ingest_ecfr
from .obs import logging as obslog
from .retrieve.hybrid import retrieve

# The demo set. Answers for these are generated offline by `citeline precompute`
# so the public page can serve them instantly on a host with no GPU.
#
# Chosen to show the range rather than to flatter: numeric limits, a procedural
# deadline, a long form requirement, and three questions the corpus cannot
# answer. The last two out of scope questions matter most. Benzene IS in the
# corpus, with a drinking water limit, so asking for its AIR standard is a near
# miss that a keyword filter would fail. PFOA is a real drinking water
# contaminant that is genuinely absent from this corpus edition, so a system
# that pattern matches on "MCL" and "drinking water" would happily invent one.
DEMO_QUESTIONS: tuple[str, ...] = (
    "What is the maximum contaminant level for arsenic in drinking water?",
    "At what concentration is the lead action level exceeded?",
    "What is the maximum contaminant level for total trihalomethanes?",
    "What is the maximum residual disinfectant level for chlorine?",
    "What is the MCL for combined radium-226 and radium-228?",
    "How quickly must a system issue a Tier 1 public notice?",
    "What must a community water system include in its consumer confidence report?",
    "What is the capital of France?",
    "What is the national ambient air quality standard for benzene?",
    "What is the drinking water MCL for PFOA?",
)

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


@app.command()
def precompute(
    questions_file: str = typer.Option(
        "", "--file", help="One question per line. Defaults to the built in demo set."
    ),
    force: bool = typer.Option(False, "--force", help="Regenerate answers that already exist."),
) -> None:
    """Answer the demo questions offline and store the results.

    Run this on the deployment host after an ingest. It is slow by design: the
    whole point is that the minutes are spent here, once, rather than in front
    of somebody trying the demo. Progress is printed per question so a long run
    is legible rather than silent.
    """

    # Read outside the coroutine: this is operator input read once, and doing
    # file I/O inside an async function is exactly the blocking call the linter
    # is right to object to.
    if questions_file:
        text = Path(questions_file).read_text(encoding="utf-8")
        requested = [
            line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
        ]
    else:
        requested = list(DEMO_QUESTIONS)

    async def go() -> None:
        from .generate import precomputed as pre

        qs = requested

        existing = pre.load(force=True)
        stats = await db.corpus_stats()
        corpus_version = str(stats.get("corpus_version") or "")

        answers: list[dict] = []
        for i, q in enumerate(qs, 1):
            key = pre.normalise(q)
            if not force and key in existing:
                typer.echo(f"[{i}/{len(qs)}] cached   {q}")
                answers.append(existing[key])
                continue
            typer.echo(f"[{i}/{len(qs)}] answering {q}")
            result = await answer_question(q)
            d = result.as_dict()
            typer.echo(
                f"          {'abstained' if d['abstained'] else 'answered'}"
                f" in {d['generate_ms']}ms"
                f" with {len(d['citations'])} citation(s)"
            )
            answers.append(d)

        path = pre.save(answers, corpus_version)
        answered = sum(1 for a in answers if not a["abstained"])
        typer.echo(
            json.dumps(
                {
                    "wrote": str(path),
                    "questions": len(answers),
                    "answered": answered,
                    "abstained": len(answers) - answered,
                    "corpus_version": corpus_version,
                },
                indent=2,
            )
        )
        await db.close()

    _run(go())
