-- citeline schema. Postgres 16 + pgvector.
-- Design notes:
--   * documents  : one row per source document (a regulation section).
--   * chunks     : retrieval units. content_sha is the idempotency key for the
--                  ingestion pipeline, so re-running ingest is a no-op unless
--                  the upstream text actually changed.
--   * ingest_runs: pipeline observability. Every run is recorded with counts and
--                  a terminal status, so a failed run is visible instead of silent.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL,          -- e.g. 'ecfr'
    source_ref    TEXT        NOT NULL,          -- e.g. '40 CFR 141.62'
    title         TEXT        NOT NULL,
    url           TEXT        NOT NULL,
    version       TEXT        NOT NULL,          -- corpus edition date, e.g. '2025-01-01'
    content_sha   CHAR(64)    NOT NULL,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_ref, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INT         NOT NULL,          -- position within the document
    content       TEXT        NOT NULL,
    content_sha   CHAR(64)    NOT NULL,
    token_estimate INT        NOT NULL,
    embedding     vector(768),                   -- nomic-embed-text
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal)
);

CREATE UNIQUE INDEX IF NOT EXISTS chunks_content_sha_idx ON chunks (content_sha);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx      ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- HNSW for approximate nearest neighbour. cosine distance to match the
-- normalised embeddings nomic-embed-text returns.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id             BIGSERIAL PRIMARY KEY,
    source         TEXT        NOT NULL,
    version        TEXT        NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    status         TEXT        NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running','succeeded','failed')),
    docs_seen      INT         NOT NULL DEFAULT 0,
    docs_written   INT         NOT NULL DEFAULT 0,
    chunks_written INT         NOT NULL DEFAULT 0,
    chunks_skipped INT         NOT NULL DEFAULT 0,
    quality_failures INT       NOT NULL DEFAULT 0,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS ingest_runs_started_idx ON ingest_runs (started_at DESC);

-- Query log. Used by the eval harness and the live /stats endpoint. No user
-- identifiers are stored: the question text, what was retrieved, and timings.
CREATE TABLE IF NOT EXISTS query_log (
    id            BIGSERIAL PRIMARY KEY,
    asked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT        NOT NULL,
    answered      BOOLEAN     NOT NULL,
    abstained     BOOLEAN     NOT NULL,
    top_score     REAL,
    citations     TEXT[]      NOT NULL DEFAULT '{}',
    retrieve_ms   INT         NOT NULL,
    generate_ms   INT         NOT NULL
);

CREATE INDEX IF NOT EXISTS query_log_asked_idx ON query_log (asked_at DESC);
