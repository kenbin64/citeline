"""Grounded answering with an enforced citation contract.

Three gates stand between a question and an answer, and any one of them can end
the request with an abstention:

Gate 1, before the model. If no retrieved passage is semantically close
enough to the question, or if no passage contains the question's terms at all,
the model is never called. This is the cheap gate and it catches the out of
scope question, which is the common case. It deliberately tests absolute
similarity rather than the fused rank score, because a dense retriever returns
its k nearest rows for any input at all and so its top rank carries almost no
information about whether the corpus covers the question.

Gate 2, the prompt. The model is told to reply INSUFFICIENT_CONTEXT when the
excerpts do not answer the question.

Gate 3, after the model. Every citation the model emitted is checked against the
excerpts it was actually given. A sentence with no citation, or a citation
pointing at an excerpt that was not supplied, is a failure of grounding, and the
whole answer is dropped rather than served with a warning. Serving an
ungrounded answer with a caveat attached is how a system trains its users to
ignore caveats.
"""

import re
import time
from dataclasses import dataclass, field

import httpx

from ..config import settings
from ..obs.logging import get
from ..obs.metrics import GENERATE_SECONDS
from ..retrieve.hybrid import Candidate
from .prompts import ABSTAIN_TEXT, INSUFFICIENT_MARKER, SYSTEM, build_user_prompt

log = get(__name__)

_CITE = re.compile(r"\[(\d{1,2})\]")
# Sentence split that does not fire on "141.62" or "No. 3".
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Citation:
    number: int
    source_ref: str
    title: str
    url: str


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    reason: str
    citations: list[Citation] = field(default_factory=list)
    considered: list[dict] = field(default_factory=list)
    top_score: float = 0.0
    max_similarity: float = 0.0
    retrieve_ms: int = 0
    generate_ms: int = 0
    model: str = ""

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "citations": [
                {"n": c.number, "ref": c.source_ref, "title": c.title, "url": c.url}
                for c in self.citations
            ],
            "considered": self.considered,
            "top_score": round(self.top_score, 5),
            "max_similarity": round(self.max_similarity, 4),
            "retrieve_ms": self.retrieve_ms,
            "generate_ms": self.generate_ms,
            "model": self.model,
        }


async def _call_ollama(system: str, user: str) -> str:
    cfg = settings()
    async with httpx.AsyncClient(base_url=cfg.ollama_url, timeout=cfg.generate_timeout_s) as c:
        r = await c.post(
            "/api/chat",
            json={
                "model": cfg.generate_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    # Temperature 0 so the same question and the same corpus give
                    # the same answer. An auditable system cannot be creative.
                    "temperature": 0.0,
                    "num_ctx": 8192,
                },
            },
        )
        r.raise_for_status()
        return str(r.json().get("message", {}).get("content", "")).strip()


def verify_citations(text: str, supplied: set[int]) -> tuple[bool, str, set[int]]:
    """Check that the answer is actually grounded in the excerpts supplied.

    Returns (ok, reason, cited_numbers).
    """
    cited = {int(m) for m in _CITE.findall(text)}

    if not cited:
        return False, "answer contained no citation", cited

    invented = cited - supplied
    if invented:
        return False, f"answer cited excerpts that were not supplied: {sorted(invented)}", cited

    # Every substantive sentence needs its own citation. The split runs over the
    # original text so each sentence still carries its brackets. Short
    # connective fragments are exempt so the model can write readable prose.
    for sentence in _SENTENCE.split(text):
        s = sentence.strip()
        if not s:
            continue
        # Measure length without the citation markers, so "[1][2]" does not make
        # a two word fragment look substantive.
        if len(_CITE.sub("", s).strip()) < 40:
            continue
        if not _CITE.search(s):
            return False, "a substantive sentence carried no citation", cited

    return True, "grounded", cited


async def answer_question(question: str) -> Answer:
    from ..retrieve.hybrid import retrieve  # local import keeps the import graph acyclic

    cfg = settings()
    result = await retrieve(question)
    considered = [
        {
            "ref": c.source_ref,
            "title": c.title,
            "score": round(c.score, 5),
            "similarity": round(c.similarity, 4) if c.similarity is not None else None,
            "found_by": c.found_by,
        }
        for c in result.candidates
    ]

    # --- Gate 1: retrieval confidence, before spending a single token ---
    gate_reason: str | None = None
    if len(result.candidates) < cfg.min_supporting_chunks:
        gate_reason = "retrieval returned nothing"
    elif result.max_similarity < cfg.min_similarity:
        gate_reason = (
            f"best passage similarity {result.max_similarity:.3f} is below the "
            f"{cfg.min_similarity} threshold, so the corpus does not appear to "
            f"cover this question"
        )
    elif cfg.require_lexical_match and not result.lexical_matched:
        gate_reason = (
            "no passage contained the question's terms, so the wording does not "
            "appear anywhere in the corpus"
        )

    if gate_reason:
        log.info(
            "answer.abstained",
            gate="retrieval",
            max_similarity=round(result.max_similarity, 4),
            threshold=cfg.min_similarity,
            lexical_hits=result.lexical_hits,
        )
        return Answer(
            question=question,
            text=ABSTAIN_TEXT,
            abstained=True,
            reason=gate_reason,
            considered=considered,
            top_score=result.top_score,
            max_similarity=result.max_similarity,
            retrieve_ms=result.retrieve_ms,
            model=cfg.generate_model,
        )

    numbered: list[tuple[int, str, str]] = [
        (i, c.source_ref, c.content) for i, c in enumerate(result.candidates, start=1)
    ]
    by_number: dict[int, Candidate] = dict(enumerate(result.candidates, start=1))

    started = time.perf_counter()
    raw = await _call_ollama(SYSTEM, build_user_prompt(question, numbered))
    generate_ms = round((time.perf_counter() - started) * 1000)
    GENERATE_SECONDS.observe(generate_ms / 1000)

    # --- Gate 2: the model declined ---
    if INSUFFICIENT_MARKER in raw.upper():
        log.info("answer.abstained", gate="model")
        return Answer(
            question=question,
            text=ABSTAIN_TEXT,
            abstained=True,
            reason="model reported the excerpts do not answer the question",
            considered=considered,
            top_score=result.top_score,
            max_similarity=result.max_similarity,
            retrieve_ms=result.retrieve_ms,
            generate_ms=generate_ms,
            model=cfg.generate_model,
        )

    # --- Gate 3: verify the citation contract ---
    ok, reason, cited = verify_citations(raw, set(by_number))
    if not ok:
        log.info("answer.abstained", gate="verification", detail=reason)
        return Answer(
            question=question,
            text=ABSTAIN_TEXT,
            abstained=True,
            reason=f"failed citation verification: {reason}",
            considered=considered,
            top_score=result.top_score,
            max_similarity=result.max_similarity,
            retrieve_ms=result.retrieve_ms,
            generate_ms=generate_ms,
            model=cfg.generate_model,
        )

    citations = [
        Citation(
            number=n,
            source_ref=by_number[n].source_ref,
            title=by_number[n].title,
            url=by_number[n].url,
        )
        for n in sorted(cited)
    ]

    log.info(
        "answer.served",
        citations=len(citations),
        top_score=round(result.top_score, 5),
        generate_ms=generate_ms,
    )
    return Answer(
        question=question,
        text=raw,
        abstained=False,
        reason=reason,
        citations=citations,
        considered=considered,
        top_score=result.top_score,
        retrieve_ms=result.retrieve_ms,
        generate_ms=generate_ms,
        model=cfg.generate_model,
    )
