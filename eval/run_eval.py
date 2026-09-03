"""Evaluation harness.

It measures two different things, because a RAG system fails in two different
ways and a single number hides one of them.

RETRIEVAL, on the answerable questions only:
  recall@k   did the gold section appear anywhere in the top k
  hit@1      was the gold section ranked first
  MRR        1 / rank of the first gold section, averaged
  latency    p50 and p95 milliseconds

ANSWERING, on every question:
  answer accuracy    on answerable questions, did the served answer contain the
                     required fact and cite the gold section
  abstention recall  on out of scope questions, did it refuse
  false answer rate  out of scope questions that got an answer anyway. This is
                     the number that matters. It is the hallucination rate.

Deliberately no LLM as judge. The gold set asserts specific strings that are
either present or absent, so the score is reproducible and cannot drift with
the mood of a grader model.

Usage:
    python eval/run_eval.py                      # retrieval and answering
    python eval/run_eval.py --retrieval-only     # fast, no generation
    python eval/run_eval.py --out eval/report.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citeline import db  # noqa: E402
from citeline.config import settings  # noqa: E402
from citeline.generate.answer import answer_question  # noqa: E402
from citeline.obs import logging as obslog  # noqa: E402
from citeline.retrieve.hybrid import retrieve  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
K = 6


@dataclass
class Case:
    id: str
    question: str
    kind: str
    gold_refs: list[str]
    must_contain: list[str]
    note: str = ""


@dataclass
class RetrievalRow:
    id: str
    question: str
    gold_refs: list[str]
    retrieved: list[str]
    rank: int | None
    hit_at_1: bool
    recall_at_k: bool
    reciprocal_rank: float
    top_score: float
    max_similarity: float
    lexical_hits: int
    retrieve_ms: int


@dataclass
class AnswerRow:
    id: str
    question: str
    kind: str
    abstained: bool
    reason: str
    answer: str
    citations: list[str]
    facts_present: bool
    cited_gold: bool
    correct: bool
    generate_ms: int


@dataclass
class Report:
    corpus: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    answering: dict = field(default_factory=dict)
    retrieval_rows: list[dict] = field(default_factory=list)
    answer_rows: list[dict] = field(default_factory=list)
    generated_at: str = ""


def load_cases() -> list[Case]:
    cases = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(Case(**json.loads(line)))
    return cases


async def eval_retrieval(cases: list[Case]) -> tuple[dict, list[RetrievalRow]]:
    answerable = [c for c in cases if c.kind == "answerable"]
    rows: list[RetrievalRow] = []

    for c in answerable:
        result = await retrieve(c.question, final_k=K)
        refs = [x.source_ref for x in result.candidates]
        rank = next((i for i, r in enumerate(refs, start=1) if r in c.gold_refs), None)
        rows.append(
            RetrievalRow(
                id=c.id,
                question=c.question,
                gold_refs=c.gold_refs,
                retrieved=refs,
                rank=rank,
                hit_at_1=rank == 1,
                recall_at_k=rank is not None,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                top_score=round(result.top_score, 5),
                max_similarity=round(result.max_similarity, 4),
                lexical_hits=result.lexical_hits,
                retrieve_ms=result.retrieve_ms,
            )
        )

    lat = sorted(r.retrieve_ms for r in rows)
    n = len(rows) or 1
    summary = {
        "questions": len(rows),
        "k": K,
        "recall_at_k": round(sum(r.recall_at_k for r in rows) / n, 4),
        "hit_at_1": round(sum(r.hit_at_1 for r in rows) / n, 4),
        "mrr": round(sum(r.reciprocal_rank for r in rows) / n, 4),
        "p50_retrieve_ms": lat[len(lat) // 2] if lat else None,
        "p95_retrieve_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "mean_retrieve_ms": round(statistics.fmean(lat)) if lat else None,
    }
    return summary, rows


async def eval_answering(cases: list[Case]) -> tuple[dict, list[AnswerRow]]:
    rows: list[AnswerRow] = []

    for c in cases:
        a = await answer_question(c.question)
        cited = [x.source_ref for x in a.citations]
        facts_present = all(f.lower() in a.text.lower() for f in c.must_contain)
        cited_gold = bool(set(cited) & set(c.gold_refs)) if c.gold_refs else False

        if c.kind == "answerable":
            correct = (not a.abstained) and facts_present and cited_gold
        else:
            correct = a.abstained

        rows.append(
            AnswerRow(
                id=c.id,
                question=c.question,
                kind=c.kind,
                abstained=a.abstained,
                reason=a.reason,
                answer=a.text,
                citations=cited,
                facts_present=facts_present,
                cited_gold=cited_gold,
                correct=correct,
                generate_ms=a.generate_ms,
            )
        )
        print(
            f"  {c.id:9s} {'ABSTAIN' if a.abstained else 'ANSWER ':8s} "
            f"{'PASS' if correct else 'FAIL'}  {c.question[:52]}",
            flush=True,
        )

    ans = [r for r in rows if r.kind == "answerable"]
    oos = [r for r in rows if r.kind == "out_of_scope"]
    na, no = len(ans) or 1, len(oos) or 1

    summary = {
        "answerable": {
            "questions": len(ans),
            "correct": sum(r.correct for r in ans),
            "accuracy": round(sum(r.correct for r in ans) / na, 4),
            "abstained_wrongly": sum(r.abstained for r in ans),
            "cited_gold_section": round(sum(r.cited_gold for r in ans) / na, 4),
        },
        "out_of_scope": {
            "questions": len(oos),
            "abstained": sum(r.abstained for r in oos),
            "abstention_recall": round(sum(r.abstained for r in oos) / no, 4),
            "false_answer_rate": round(sum(not r.abstained for r in oos) / no, 4),
            "false_answers": [r.id for r in oos if not r.abstained],
        },
        "overall_correct": round(sum(r.correct for r in rows) / (len(rows) or 1), 4),
        "mean_generate_ms": round(statistics.fmean([r.generate_ms for r in rows if r.generate_ms]))
        if any(r.generate_ms for r in rows)
        else None,
    }
    return summary, rows


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.json"))
    args = p.parse_args()

    obslog.configure("WARNING")
    cases = load_cases()
    cfg = settings()
    report = Report(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        corpus=await db.corpus_stats(),
        config={
            "embed_model": cfg.embed_model,
            "generate_model": cfg.generate_model,
            "vector_k": cfg.vector_k,
            "lexical_k": cfg.lexical_k,
            "rrf_k": cfg.rrf_k,
            "final_k": cfg.final_k,
            "min_similarity": cfg.min_similarity,
            "require_lexical_match": cfg.require_lexical_match,
        },
    )

    print(f"citeline eval  |  {len(cases)} cases  |  corpus: {report.corpus}")
    print("\nRetrieval")
    rsum, rrows = await eval_retrieval(cases)
    report.retrieval = rsum
    report.retrieval_rows = [asdict(r) for r in rrows]
    print(json.dumps(rsum, indent=2))

    if not args.retrieval_only:
        print("\nAnswering")
        asum, arows = await eval_answering(cases)
        report.answering = asum
        report.answer_rows = [asdict(r) for r in arows]
        print(json.dumps(asum, indent=2))

    Path(args.out).write_text(
        json.dumps(asdict(report), indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
