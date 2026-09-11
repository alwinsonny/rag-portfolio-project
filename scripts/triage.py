"""Why did a question miss? Separate bad questions from bad retrieval.

    python scripts/triage.py                  # every question that missed
    python scripts/triage.py --id g-003       # one question

WHY THIS MATTERS BEFORE PHASE 3:
A miss has two very different causes, and they need opposite responses.

  BAD RETRIEVAL   The labelled passage is findable — searching its own text
                  returns it at rank 1 — but the question does not reach it.
                  This is a real retrieval failure and Phase 3 should fix it.

  BAD QUESTION    The question is too generic ("side effects of schizophrenia"
                  matches dozens of passages) or too long and multi-clause (one
                  embedding averaged across four clauses becomes mush). The
                  system is behaving correctly; the label is the problem.

Counting these separately keeps you honest. If most of your misses are bad
questions, your true baseline is higher than it looks, and improving retrieval
would be fixing a problem you do not have.

For each miss this prints:
  * whether the labelled chunk is findable by its own words (a sanity check —
    if it is not, something is wrong upstream of everything else)
  * what retrieval actually returned instead
  * the labelled passage, so you can judge whether the question was fair
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from drug_label_rag.db.models import Chunk
from drug_label_rag.db.session import dispose, session_scope
from drug_label_rag.retrieval.dense import dense_search

GOLDEN = Path("data/golden.jsonl")


def load() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def triage_one(session, row: dict, k: int) -> str:
    question = row["question"]
    target_ids = [r["chunk_id"] for r in row["relevant"]]

    target = (
        await session.execute(
            select(Chunk.content, Chunk.generic_name, Chunk.section).where(
                Chunk.id == target_ids[0]
            )
        )
    ).first()
    if target is None:
        return "TARGET MISSING — chunk id no longer exists. Run scripts/resolve.py."

    hits = await dense_search(session, question, k=k)
    rank = next((i for i, h in enumerate(hits, 1) if h.chunk_id in target_ids), None)

    print("\n" + "=" * 76)
    print(f"[{row['id']}]  {row.get('type', '?')}")
    print(f"Q: {question}")
    print("=" * 76)

    words = len(question.split())
    clauses = question.count(",") + question.count("(") + 1

    if rank:
        print(f"  FOUND at rank {rank} of {k}")
        verdict = f"found at {rank}"
    else:
        # Sanity check: can the labelled passage be retrieved by its own words?
        anchor = row.get("anchor_text") or " ".join(target.content.split()[:12])
        self_hits = await dense_search(session, anchor, k=10)
        self_rank = next(
            (i for i, h in enumerate(self_hits, 1) if h.chunk_id in target_ids), None
        )
        if self_rank is None:
            print("  ⚠ The labelled passage is NOT retrievable even by its own text.")
            print("    Something is wrong upstream — check embeddings and the index.")
            verdict = "INDEX PROBLEM"
        elif words > 22 or clauses >= 3:
            print(f"  MISS — question is {words} words, ~{clauses} clauses.")
            print("    A single embedding averaged across several clauses loses focus.")
            print("    Split it into one question per fact.")
            verdict = "bad question: too long"
        else:
            print(f"  MISS — passage is findable (self-search rank {self_rank}),")
            print("    so the question simply does not reach it. Genuine retrieval gap.")
            verdict = "RETRIEVAL GAP"

    print(f"\n  LABELLED PASSAGE  ({target.generic_name} / {target.section}):")
    print(f"    {' '.join(target.content.split())[:320]}")

    print("\n  RETRIEVAL RETURNED:")
    for i, hit in enumerate(hits[:3], 1):
        mark = "->" if hit.chunk_id in target_ids else "  "
        print(f"  {mark}[{i}] {hit.score:.3f} {hit.citation}")
        print(f"        {' '.join(hit.content.split())[:150]}")

    return verdict


async def run(only: str | None, k: int) -> None:
    rows = load()
    if only:
        rows = [r for r in rows if r["id"] == only]

    verdicts: list[tuple[str, str]] = []
    async with session_scope() as session:
        for row in rows:
            hits = await dense_search(session, row["question"], k=k)
            targets = {r["chunk_id"] for r in row["relevant"]}
            if not only and any(h.chunk_id in targets for h in hits[:10]):
                continue  # only triage the misses
            verdicts.append((row["id"], await triage_one(session, row, k)))

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    counts: dict[str, int] = {}
    for qid, verdict in verdicts:
        key = verdict.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        print(f"  {qid}  {verdict}")
    print()
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {key}")
    print("\nRETRIEVAL GAP  -> Phase 3 should fix these.")
    print("bad question   -> rewrite or drop; they are not measuring retrieval.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose golden-set misses.")
    parser.add_argument("--id", help="Triage one question")
    parser.add_argument("-k", type=int, default=50)
    args = parser.parse_args(argv)

    async def _main() -> None:
        try:
            await run(args.id, args.k)
        finally:
            await dispose()

    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())