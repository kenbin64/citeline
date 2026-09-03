"""Runtime configuration. Everything is environment driven so the same image runs
locally and on the server with no code change."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITELINE_", env_file=".env", extra="ignore")

    # --- storage ---
    database_url: str = "postgresql://citeline:citeline@127.0.0.1:5432/citeline"
    db_pool_min: int = 1
    db_pool_max: int = 8

    # --- models (Ollama, local, no vendor API key) ---
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # Chosen by measurement on this host, which is a QEMU virtual CPU with no
    # AVX or SSE4.2. The 8B model took roughly 150s per answer there; the 3B
    # model follows the citation format as well and is several times faster.
    generate_model: str = "llama3.2:3b"
    generate_timeout_s: float = 120.0

    # --- chunking ---
    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 60
    chunk_min_chars: int = 120

    # --- retrieval ---
    vector_k: int = 24
    lexical_k: int = 24
    rrf_k: int = 60
    # Kept small on purpose: prompt prefill dominates latency on a CPU only
    # host, and 4 whole regulation sections is already more context than most
    # questions need.
    final_k: int = 4

    # --- the abstention gate ---
    # The service refuses to answer unless retrieval clears these bars. This is
    # the whole point of the product, so the thresholds are first class settings
    # and every decision is logged with the numbers that produced it.
    #
    # The gate is on raw cosine similarity, NOT on the fused RRF score. RRF is a
    # rank statistic: a dense retriever returns its k nearest rows for any input
    # whatsoever, so the top fused score barely moves between a question the
    # corpus answers and one it has never heard of. Absolute similarity does move.
    #
    # min_similarity was set from measurement, not taste. See eval/thresholds.md
    # for the separation between in scope and out of scope questions.
    min_similarity: float = 0.62
    # A question sharing no vocabulary with the corpus produces zero full text
    # matches. Requiring a lexical match as well is a second, independent signal,
    # and the two together are what push the false answer rate down.
    require_lexical_match: bool = True
    min_supporting_chunks: int = 1

    # --- api ---
    host: str = "127.0.0.1"
    port: int = 8811
    cors_origins: list[str] = ["https://butterflyfx.us", "http://localhost:8000"]
    rate_limit_per_minute: int = 20
    max_question_chars: int = 500

    # --- corpus ---
    ecfr_title: int = 40
    ecfr_part: int = 141
    ecfr_version: str = "2025-01-01"


@lru_cache
def settings() -> Settings:
    return Settings()
