"""Request and response models. Pydantic gives the API a typed contract and a
generated OpenAPI schema, which is what lets the portfolio page and the eval
harness talk to the same endpoint without drifting apart."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be blank")
        return v


class CitationOut(BaseModel):
    n: int
    ref: str
    title: str
    url: str


class ConsideredOut(BaseModel):
    ref: str
    title: str
    score: float
    similarity: float | None = None
    found_by: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    reason: str
    citations: list[CitationOut]
    considered: list[ConsideredOut]
    top_score: float
    max_similarity: float
    retrieve_ms: int
    generate_ms: int
    model: str
    # True when this answer was generated ahead of time rather than on the spot.
    # Stated in the response on purpose: a stored answer presented as a live one
    # would be its own small dishonesty in a project about not bluffing.
    precomputed: bool = False


class RetrieveResponse(BaseModel):
    """Retrieval only, no generation. Fast, and it is the honest way to show what
    the retriever did without waiting on a CPU bound language model."""

    question: str
    candidates: list[ConsideredOut]
    top_score: float
    max_similarity: float
    retrieve_ms: int
    vector_hits: int
    lexical_hits: int
    passed_gate: bool
    threshold: float
    gate_reason: str


class HealthResponse(BaseModel):
    status: str
    database: bool
    embedding_model: bool
    generation_model: bool
    documents: int
    chunks: int
    corpus_version: str | None
    last_ingest: str | None


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    unembedded: int
    corpus_version: str | None
    last_ingest: str | None
    queries_total: int
    abstention_rate: float
    median_retrieve_ms: int | None
