"""Embeddings via a local Ollama instance.

Deliberate choice: no vendor API. The whole corpus is embedded on the server's
CPU, which means no key to leak, no per token cost, no data leaving the box, and
a reproducible index. The tradeoff is throughput, which is why batches are
concurrent and the pipeline is resumable.
"""

import asyncio
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings
from ..obs.logging import get
from ..obs.metrics import EMBED_SECONDS

log = get(__name__)


class EmbeddingError(RuntimeError):
    pass


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, EmbeddingError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _embed_one(client: httpx.AsyncClient, text: str, model: str) -> list[float]:
    r = await client.post("/api/embeddings", json={"model": model, "prompt": text})
    r.raise_for_status()
    vec = r.json().get("embedding")
    if not vec:
        raise EmbeddingError("ollama returned no embedding")
    return vec


async def embed_texts(texts: list[str], concurrency: int = 4) -> list[list[float]]:
    """Embed a list of texts, preserving order."""
    cfg = settings()
    if not texts:
        return []
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=180.0) as client:

        async def one(t: str) -> list[float]:
            async with sem:
                return await _embed_one(client, t, cfg.embed_model)

        vectors = await asyncio.gather(*(one(t) for t in texts))

    elapsed = time.perf_counter() - started
    EMBED_SECONDS.observe(elapsed)

    bad = [i for i, v in enumerate(vectors) if len(v) != cfg.embed_dim]
    if bad:
        raise EmbeddingError(
            f"embedding dimension mismatch at {bad[:3]}: "
            f"expected {cfg.embed_dim}, model {cfg.embed_model}"
        )

    log.info("embed.batch", n=len(texts), seconds=round(elapsed, 3))
    return list(vectors)


async def embed_documents(texts: list[str], concurrency: int = 4) -> list[list[float]]:
    """Embed passages for storage.

    nomic-embed-text is trained with task prefixes and expects both sides to be
    labelled: passages as search_document, questions as search_query. Prefixing
    only one side is worse than prefixing neither, because the two vector spaces
    end up systematically offset from each other.
    """
    return await embed_texts([f"search_document: {t}" for t in texts], concurrency=concurrency)


async def embed_query(text: str) -> list[float]:
    """Embed a single question."""
    vectors = await embed_texts([f"search_query: {text}"], concurrency=1)
    return vectors[0]


def to_pgvector(vec: list[float]) -> str:
    """asyncpg has no native vector codec, so send the pgvector text form."""
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"
