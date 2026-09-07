"""Answers computed ahead of time, so the public demo does not generate on demand.

Why this exists, measured rather than assumed: on the deployment host (16 CPU
cores, no GPU) a single grounded answer over a realistic prompt ran past four
minutes. A public demo that takes four minutes is a broken demo, and the honest
options were a GPU host, a smaller model that answers worse, or not generating
on demand at all.

This is the third one. Generation happens once, offline, through exactly the
same pipeline the service would use live, and the result is stored. The demo
then serves it instantly.

Two rules keep this from being a fake:

1. A precomputed answer is LABELLED as one in the response. Nothing here
   pretends a stored answer was produced on the spot.
2. Retrieval is never precomputed. Ask any question and the retriever runs live
   against the real index, because retrieval is fast and is the part worth
   showing interactively.

The store is a plain JSON file rather than a database table. It is small, it is
read once at startup, it belongs to the deployment rather than the corpus, and
keeping it out of Postgres means the demo still works against an empty database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import settings
from ..obs import logging as obslog

log = obslog.get(__name__)

_cache: dict[str, dict[str, Any]] | None = None


def normalise(question: str) -> str:
    """A forgiving key, so trailing punctuation or spacing does not miss a hit.

    Deliberately not fuzzy matching. A near miss returning some other question's
    answer would be exactly the confident wrong answer this project exists to
    avoid, so the match is exact once case, punctuation and whitespace are
    normalised, and anything else is treated as a question we do not have.
    """
    text = question.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def store_path() -> Path:
    cfg = settings()
    configured = getattr(cfg, "precomputed_path", "") or ""
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "data" / "precomputed.json"


def load(force: bool = False) -> dict[str, dict[str, Any]]:
    """Read the store. Missing or unreadable means "no precomputed answers",
    never an error: the service must still start and serve retrieval."""
    global _cache
    if _cache is not None and not force:
        return _cache
    path = store_path()
    entries: dict[str, dict[str, Any]] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw.get("answers", []):
            q = item.get("question")
            if not q:
                continue
            entries[normalise(q)] = item
        log.info("precomputed.loaded", count=str(len(entries)), path=str(path))
    except FileNotFoundError:
        log.info("precomputed.absent", path=str(path))
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the service
        log.warning("precomputed.unreadable", path=str(path), error=str(exc))
    _cache = entries
    return entries


def get(question: str) -> dict[str, Any] | None:
    return load().get(normalise(question))


def questions() -> list[str]:
    """The curated list, in the order it was written, for the demo page."""
    return [item["question"] for item in load().values()]


def save(answers: list[dict[str, Any]], corpus_version: str) -> Path:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_version": corpus_version,
        "note": (
            "Generated offline by `citeline precompute`, through the same "
            "pipeline the live service uses. Retrieval in the demo is always live."
        ),
        "answers": answers,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    load(force=True)
    return path
