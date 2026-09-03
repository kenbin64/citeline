"""Pick the abstention threshold from measurement.

The gate has one number in it, min_similarity, and that number decides whether
the product works. Set it too low and out of scope questions get confident
answers. Set it too high and real questions get refused, which is just as
useless. So it is chosen by sweeping it against the labelled set and reading off
the separation, rather than by picking something that sounds right.

The output is a table you can put in front of someone: for each candidate
threshold, how many answerable questions survive the gate and how many out of
scope questions are correctly stopped. It also prints the raw similarity for
every question, which is the part that shows whether the two classes separate at
all. If they overlap, no threshold saves you and the retrieval needs fixing
first. That is worth knowing before shipping a number.

    python eval/sweep_gate.py
    python eval/sweep_gate.py --out eval/thresholds.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citeline import db  # noqa: E402
from citeline.config import settings  # noqa: E402
from citeline.obs import logging as obslog  # noqa: E402
from citeline.retrieve.hybrid import retrieve  # noqa: E402

from run_eval import load_cases  # noqa: E402


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "thresholds.json"))
    args = p.parse_args()

    obslog.configure("WARNING")
    cfg = settings()
    cases = load_cases()

    measured = []
    print(f"Measuring {len(cases)} questions against the live index\n")
    print(f"{'id':10s} {'kind':13s} {'max_sim':>8s} {'lex':>4s}  top passage")
    print("-" * 92)

    for c in cases:
        r = await retrieve(c.question)
        top = r.candidates[0].source_ref if r.candidates else "(none)"
        measured.append(
            {
                "id": c.id,
                "kind": c.kind,
                "question": c.question,
                "max_similarity": round(r.max_similarity, 4),
                "lexical_hits": r.lexical_hits,
                "top_ref": top,
                "gold_refs": c.gold_refs,
                "gold_retrieved": bool(
                    c.gold_refs and set(x.source_ref for x in r.candidates) & set(c.gold_refs)
                ),
            }
        )
        print(
            f"{c.id:10s} {c.kind:13s} {r.max_similarity:8.4f} {r.lexical_hits:4d}  {top}",
            flush=True,
        )

    answerable = [m for m in measured if m["kind"] == "answerable"]
    oos = [m for m in measured if m["kind"] == "out_of_scope"]

    a_sims = sorted(m["max_similarity"] for m in answerable)
    o_sims = sorted(m["max_similarity"] for m in oos)
    print("\nSimilarity ranges")
    print(f"  answerable   : min {a_sims[0]:.4f}  max {a_sims[-1]:.4f}")
    print(f"  out of scope : min {o_sims[0]:.4f}  max {o_sims[-1]:.4f}")
    separated = a_sims[0] > o_sims[-1]
    print(
        f"  cleanly separated: {'YES' if separated else 'NO, the classes overlap'}"
        + ("" if separated else "  (a single threshold cannot be perfect)")
    )

    print("\nThreshold sweep  (lexical requirement applied as configured)")
    print(
        f"{'threshold':>10s} {'answerable kept':>16s} {'oos blocked':>12s} "
        f"{'false answers':>14s} {'wrongly refused':>16s}"
    )
    print("-" * 76)

    rows = []
    lo = min(a_sims[0], o_sims[0]) - 0.02
    hi = max(a_sims[-1], o_sims[-1]) + 0.02
    steps = 30
    for i in range(steps + 1):
        t = round(lo + (hi - lo) * i / steps, 4)

        def passes(m: dict, t: float = t) -> bool:
            if m["max_similarity"] < t:
                return False
            return not (cfg.require_lexical_match and m["lexical_hits"] == 0)

        kept = sum(passes(m) for m in answerable)
        blocked = sum(not passes(m) for m in oos)
        rows.append(
            {
                "threshold": t,
                "answerable_kept": kept,
                "answerable_total": len(answerable),
                "oos_blocked": blocked,
                "oos_total": len(oos),
                "false_answers": len(oos) - blocked,
                "wrongly_refused": len(answerable) - kept,
            }
        )
        print(
            f"{t:10.4f} {kept:>10d}/{len(answerable):<5d} {blocked:>7d}/{len(oos):<4d} "
            f"{len(oos) - blocked:>14d} {len(answerable) - kept:>16d}"
        )

    # Best threshold: block every out of scope question, then keep as many
    # answerable ones as possible. The asymmetry is deliberate. A wrong answer
    # to a question the corpus cannot answer destroys trust in every other
    # answer; a refusal on a question it can answer is a visible, recoverable
    # annoyance. So false answers are treated as the more expensive error.
    perfect = [r for r in rows if r["false_answers"] == 0]
    if perfect:
        best = max(perfect, key=lambda r: (r["answerable_kept"], -r["threshold"]))
        note = "blocks every out of scope question"
    else:
        best = max(rows, key=lambda r: (r["oos_blocked"] + r["answerable_kept"]))
        note = "best total accuracy; no threshold blocks everything"

    print(f"\nRecommended min_similarity = {best['threshold']}  ({note})")
    print(
        f"  answerable kept  {best['answerable_kept']}/{best['answerable_total']}\n"
        f"  out of scope blocked {best['oos_blocked']}/{best['oos_total']}\n"
        f"  currently configured: {cfg.min_similarity}"
    )

    Path(args.out).write_text(
        json.dumps(
            {
                "config": {
                    "embed_model": cfg.embed_model,
                    "require_lexical_match": cfg.require_lexical_match,
                    "configured_min_similarity": cfg.min_similarity,
                },
                "corpus": await db.corpus_stats(),
                "separated": separated,
                "recommended": best,
                "sweep": rows,
                "measured": measured,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    await db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
