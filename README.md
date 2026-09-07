# citeline

A retrieval-augmented question answering service over federal drinking water
regulations that **refuses to answer when the source text does not support one**.

Live: [`butterflyfx.us/api/citeline/healthz`](https://butterflyfx.us/api/citeline/healthz)
· [`/stats`](https://butterflyfx.us/api/citeline/stats)

```
POST /query  {"question": "What is the capital of France?"}

{
  "answer": "I do not have a sourced answer to that. The indexed regulations
             do not contain a passage that answers it, so rather than guess,
             this returns nothing.",
  "abstained": true,
  "reason": "best passage similarity 0.618 is below the 0.62 threshold, so the
             corpus does not appear to cover this question",
  "citations": [],
  "considered": [
    {"ref": "40 CFR 141.852", "score": 0.01639, "similarity": 0.6176, "found_by": "vector"},
    {"ref": "40 CFR 141.35",  "score": 0.01587, "similarity": 0.6010, "found_by": "vector"}
  ]
}
```

That is a real response from the deployed service, not an illustration.

## The problem

A RAG system that always produces an answer is worse than useless in regulated
work, because the confident wrong answer is indistinguishable from the correct
one until somebody checks. In a compliance setting the person checking is an
inspector, and by then the answer has already been acted on.

So the interesting engineering question is not "can it answer" but **"does it
know when it cannot"**, and can it prove where an answer came from.

## What it does

- Ingests a regulatory corpus (currently 40 CFR 141, the National Primary
  Drinking Water Regulations) from the public eCFR API, pinned to a dated
  corpus edition so answers are reproducible.
- Chunks by regulatory structure rather than by fixed character count, so a
  citation refers to a real section a reader can look up.
- **Hybrid retrieval**: pgvector HNSW cosine similarity for meaning, plus
  Postgres full text search for exact terms like `0.010 mg/L` or `141.62`,
  combined with Reciprocal Rank Fusion. Vector search alone misses literal
  numeric thresholds, which is most of what a regulation is.
- **Abstention gate**: when the best passage is not similar enough to the
  question, it returns no answer and says why, including what it looked at.
- **Citation contract**: the answer must cite the numbered passages it was
  given. A citation to a passage that was never retrieved is rejected rather
  than returned, so the model cannot invent a section number.
- Runs entirely on local models through Ollama. No API keys, no per token
  cost, and no corpus text leaves the host.

## Proof

Nothing here asks you to take a claim on faith.

| | |
|---|---|
| Unit tests | 31, covering chunking, fusion, the citation contract, and ingest quality |
| CI | ruff, mypy, pytest, and the schema migration applied to a real pgvector database |
| Eval set | 28 cases: 20 answerable with gold citations, 8 deliberately out of scope |
| Indexed | 185 documents, 908 chunks, 0 unembedded |

The 8 out-of-scope cases are the point. They measure whether the system stays
quiet on questions the corpus does not cover, which is the behaviour that makes
the other 20 worth anything.

```bash
make eval          # retrieval and answering metrics against eval/dataset.jsonl
make sweep         # abstention threshold sweep, precision against coverage
```

## Run it yourself

```bash
git clone https://github.com/kenbin64/citeline
cd citeline
cp .env.example .env
docker compose up -d          # postgres + pgvector, ollama
citeline migrate              # apply the schema
citeline ingest --part 141    # pull and index the corpus
citeline serve                # http://127.0.0.1:8811
```

## How it is put together

```
  eCFR API
     |
  ingest/        fetch, parse, chunk on section boundaries, quality gate
     |
  embed/         nomic-embed-text, 768 dimensions, local
     |
  Postgres + pgvector        HNSW index, plus a tsvector column
     |
  retrieve/hybrid.py         vector search + full text, fused with RRF
     |
  generate/answer.py         abstention gate, then a cited answer,
     |                       then citation verification against what was retrieved
   FastAPI
```

Structured logs and Prometheus metrics are exposed for latency, abstention
rate, and retrieval quality.

## Honest limits

- **Generation is slow on a CPU only host.** Measured on the deployment box (16
  cores, no GPU), a single generated answer over a realistic prompt ran past
  four minutes, which is why the public demo serves retrieval and abstention
  live and does not generate on demand. Retrieval is the fast path: the service
  reports its own median retrieval latency at
  [`/stats`](https://butterflyfx.us/api/citeline/stats), so the current number
  is whatever that endpoint says rather than whatever this file claims.
- The abstention threshold is tuned on a 28 case eval set. That is enough to
  show the mechanism works and not enough to claim a general error rate.
- Abstention catches *unsupported* answers. It does not catch an answer that is
  fluent, cited, and still a misreading of the passage. No system here claims a
  hallucination rate of zero.
- One CFR part is indexed. The ingest is written to take more, but breadth has
  not been tested.

## Licence

MIT. See [LICENSE](LICENSE).
