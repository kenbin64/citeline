"""Prometheus metrics. Exposed at /metrics so retrieval latency and the
abstention rate are observable rather than anecdotal."""

from prometheus_client import Counter, Gauge, Histogram

QUERIES = Counter("citeline_queries_total", "Questions received", ["outcome"])

RETRIEVE_SECONDS = Histogram(
    "citeline_retrieve_seconds",
    "Hybrid retrieval latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

GENERATE_SECONDS = Histogram(
    "citeline_generate_seconds",
    "Grounded generation latency",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160),
)

EMBED_SECONDS = Histogram(
    "citeline_embed_seconds",
    "Embedding latency per batch",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0),
)

TOP_SCORE = Histogram(
    "citeline_top_score",
    "Fused score of the best candidate per query",
    buckets=(0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.12, 0.2),
)

INGEST_CHUNKS = Counter("citeline_ingest_chunks_total", "Chunks written by ingest", ["result"])

CORPUS_CHUNKS = Gauge("citeline_corpus_chunks", "Chunks currently indexed")
CORPUS_DOCS = Gauge("citeline_corpus_documents", "Documents currently indexed")
