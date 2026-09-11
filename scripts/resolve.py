"""Re-point golden-set labels at their new chunk ids after a re-ingest.

    python scripts/resolve.py --check     # report drift, change nothing
    python scripts/resolve.py --apply     # rewrite data/golden.jsonl

THE PROBLEM THIS SOLVES:
chunk_id is assigned at insert time. Phase 3 changes the chunking strategy,
which means a re-ingest, which means every id is reassigned. Without this
script your 150 labelled questions would all point at the wrong passages,
recall would collapse to near zero, and you would spend an evening debugging
retrieval that is working perfectly.

That failure is silent and extremely convincing. Run --check after every
re-ingest, before you run an evaluation.

HOW IT WORKS:
Each label carries an anchor_text — a short verbatim quote from the passage it
was written against. After re-chunking, the same words still exist somewhere in
the corpus, just in a differently-bounded chunk. A full-text search for the
anchor, restricted to the same document, finds the new home.

Labels written before anchor_text existed fall back to matching on set_id and
section, which is weaker: it can only narrow to the document, so it reports
ambiguity rather than guessing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from drug_label_rag.db.models import Chunk
from drug_label_rag.db.session import dispose, session_scope

GOLDEN = Path("data/golden.jsonl")


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(path: Path, rows: list[dict[str, Any]]) -> None:
    backup = path.with_suffix(".jsonl.bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"previous version kept at {backup}")


async def find_chunk(session, set_id: str | None, anchor: str | None) -> list[int]:
    """Find candidate chunk ids for one label."""
    if not anchor:
        return []
    stmt = select(Chunk.id).where(Chunk.content.contains(anchor))
    if set_id:
        stmt = stmt.where(Chunk.set_id == set_id)
    ids = [row[0] for row in (await session.execute(stmt.limit(5))).all()]
    if ids:
        return ids

    # The anchor may straddle a new chunk boundary. Retry with the first half —
    # shorter anchors match more loosely but still land in the right document.
    words = anchor.split()
    if len(words) > 6:
        short = " ".join(words[: len(words) // 2])
        stmt = select(Chunk.id).where(Chunk.content.contains(short))
        if set_id:
            stmt = stmt.where(Chunk.set_id == set_id)
        return [row[0] for row in (await session.execute(stmt.limit(5))).all()]
    return []


async def resolve(apply: bool) -> None:
    rows = load(GOLDEN)
    if not rows:
        raise SystemExit(f"{GOLDEN} is empty.")

    ok = moved = ambiguous = lost = no_anchor = 0

    async with session_scope() as session:
        live = {
            row[0] for row in (await session.execute(select(Chunk.id))).all()
        }
        total_chunks = len(live)

        for row in rows:
            anchor = row.get("anchor_text")
            if not anchor:
                no_anchor += 1
                continue

            current = [r["chunk_id"] for r in row["relevant"]]
            # If the current id still exists AND still contains the anchor,
            # nothing has moved.
            if all(cid in live for cid in current):
                still_valid = (
                    await session.execute(
                        select(func.count())
                        .select_from(Chunk)
                        .where(Chunk.id.in_(current), Chunk.content.contains(anchor))
                    )
                ).scalar_one()
                if still_valid:
                    ok += 1
                    continue

            found = await find_chunk(session, row.get("set_id"), anchor)
            if not found:
                lost += 1
                print(f"  LOST      [{row['id']}] {row['question'][:56]}")
            elif len(found) > 1:
                ambiguous += 1
                print(f"  AMBIGUOUS [{row['id']}] {len(found)} candidates: {found}")
                if apply:
                    row["relevant"] = [{"chunk_id": found[0], "grade": 2}]
            else:
                moved += 1
                if apply:
                    grade = row["relevant"][0].get("grade", 2)
                    row["relevant"] = [{"chunk_id": found[0], "grade": grade}]

    print()
    print(f"corpus holds {total_chunks} chunks")
    print(f"  {ok:>4} unchanged")
    print(f"  {moved:>4} re-pointed to a new chunk")
    print(f"  {ambiguous:>4} ambiguous (anchor matched several chunks)")
    print(f"  {lost:>4} lost (anchor not found — relabel these)")
    if no_anchor:
        print(f"  {no_anchor:>4} have no anchor_text and cannot be re-resolved")

    if apply and (moved or ambiguous):
        save(GOLDEN, rows)
        print(f"\n{GOLDEN} rewritten.")
    elif not apply:
        print("\nDry run. Re-run with --apply to rewrite the file.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-resolve golden-set labels.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Report drift only")
    group.add_argument("--apply", action="store_true", help="Rewrite the file")
    args = parser.parse_args(argv)

    async def _main() -> None:
        try:
            await resolve(apply=args.apply)
        finally:
            await dispose()

    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())