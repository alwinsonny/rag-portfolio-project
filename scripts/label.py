"""Labelling helper for the golden set.

PHASE 1 needs ~20 questions for a provisional number.
PHASE 2 grows this to 150 answerable plus 40 unanswerable.

    python scripts/label.py                      # random chunks
    python scripts/label.py --section drug_interactions
    python scripts/label.py --type named_entity
    python scripts/label.py --stats

THE ONE RULE:
You are shown a passage. Write the question THAT PASSAGE ANSWERS. Not a
question about drugs in general — a question a professional would ask whose
answer is in the text on screen. If the passage is about montelukast and
allergic rhinitis, "which medicine is used for fever" is not a label, it is
noise, and it will show up as a retrieval failure that is really a labelling
failure.

If the passage is a mid-sentence fragment you cannot write a sensible question
about, press ENTER to skip it. Under fixed-size chunking many chunks are exactly
that — and noticing how many is itself the Phase 1 finding.

WHY ANCHOR TEXT IS CAPTURED:
chunk_id is not stable. Phase 3 re-chunks the corpus and every id is reassigned,
which would invalidate the entire golden set. So each label also records a short
verbatim quote from the passage. scripts/resolve.py uses that quote to find the
label's new home after a re-ingest, so 150 questions survive a chunking change
instead of needing to be rewritten.

QUESTION TYPES (Phase 2 target distribution):
    direct_lookup   45   one fact, one section
    named_entity    25   a specific drug, dose or code — where dense fails
    paraphrase      25   few shared words with the passage — where lexical fails
    multi_passage   25   needs two chunks; label BOTH
    hard_negative   20   a similar-looking wrong passage exists
    metadata        10   implies a route, population or date constraint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from drug_label_rag.db.models import Chunk
from drug_label_rag.db.session import dispose, session_scope

GOLDEN = Path("data/golden.jsonl")
UNANSWERABLE = Path("data/unanswerable.jsonl")

TYPES = [
    "direct_lookup",
    "named_entity",
    "paraphrase",
    "multi_passage",
    "hard_negative",
    "metadata",
]

STOPWORDS = {
    "what", "which", "when", "where", "how", "why", "is", "are", "the", "a", "an",
    "of", "for", "in", "to", "with", "and", "or", "can", "do", "does", "should",
    "used", "use", "be", "on", "at", "by", "from", "this", "that", "it", "you",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def overlap_ratio(question: str, passage: str) -> float:
    """Share of the question's content words that appear in the passage.

    Two failure modes sit at opposite ends of this number:

      near 0.0  the question has nothing to do with the passage — a mislabel
      near 1.0  the question is the passage reworded, which tests paraphrase
                matching that dense retrieval already does well, and flatters
                your results
    """
    words = content_words(question)
    if not words:
        return 0.0
    return len(words & content_words(passage)) / len(words)


def anchor_from(passage: str, words: int = 12) -> str:
    """A distinctive verbatim quote used to re-find this passage after re-chunking.

    Taken from the middle rather than the start: fixed-size chunks often begin
    mid-word, and a fragment like "ck, and 4.8% Hispanic" is a poor anchor.
    """
    tokens = passage.split()
    start = max(0, len(tokens) // 2 - words // 2)
    return " ".join(tokens[start : start + words])


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def show_stats() -> None:
    rows = load_existing(GOLDEN)
    print(f"\n{len(rows)} answerable questions")
    for kind, n in Counter(r.get("type", "?") for r in rows).most_common():
        print(f"  {kind:<16} {n}")
    print(f"{len(load_existing(UNANSWERABLE))} unanswerable questions")
    print("Phase 2 targets: 150 answerable, 40 unanswerable")


async def label_loop(section: str | None, qtype: str, count: int, min_len: int) -> None:
    existing = load_existing(GOLDEN)
    used = {r["id"] for r in existing}
    next_n = len(existing) + 1

    async with session_scope() as session:
        stmt = select(
            Chunk.id, Chunk.content, Chunk.section, Chunk.set_id, Chunk.generic_name
        ).where(func.length(Chunk.content) >= min_len)
        if section:
            stmt = stmt.where(Chunk.section == section)
        rows = (
            await session.execute(stmt.order_by(func.random()).limit(count * 5))
        ).all()

    if not rows:
        raise SystemExit("No chunks matched. Have you ingested?")

    random.shuffle(rows)
    written = skipped = 0

    for row in rows:
        if written >= count:
            break
        passage = " ".join(row.content.split())
        print("\n" + "=" * 74)
        print(f"chunk {row.id}   {row.generic_name}   [{row.section}]   "
              f"{len(row.content)} chars")
        print("=" * 74)
        print(passage[:1500])
        print("-" * 74)
        print("What question does THIS PASSAGE answer?")
        print("ENTER to skip (do skip mid-sentence fragments) · 'q' to stop")

        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() == "q":
            break
        if not question:
            skipped += 1
            continue

        # Guard against the two failure modes before the label is written.
        ratio = overlap_ratio(question, passage)
        if ratio < 0.25:
            print(f"\n  ⚠  Only {ratio:.0%} of your question's words appear in this")
            print("     passage. Is the question actually answered by the text above?")
            if input("     keep anyway? [y/N]> ").strip().lower() != "y":
                skipped += 1
                continue
        elif ratio > 0.9:
            print(f"\n  ⚠  {ratio:.0%} word overlap — this reads as the passage reworded.")
            print("     That tests paraphrase matching, which dense retrieval already")
            print("     does well, and will flatter your numbers.")
            if input("     keep anyway? [y/N]> ").strip().lower() != "y":
                skipped += 1
                continue

        kind = input(f"type [{qtype}]> ").strip() or qtype
        if kind not in TYPES:
            print(f"  unknown type, using {qtype}")
            kind = qtype
        notes = input("notes (optional)> ").strip()

        relevant = [{"chunk_id": row.id, "grade": 2}]
        if kind == "multi_passage":
            second = input("second relevant chunk_id (grade 1)> ").strip()
            if second.isdigit():
                relevant.append({"chunk_id": int(second), "grade": 1})

        while f"g-{next_n:03d}" in used:
            next_n += 1
        record = {
            "id": f"g-{next_n:03d}",
            "type": kind,
            "question": question,
            "relevant": relevant,
            "set_id": row.set_id,
            "section": row.section,
            # Survives re-chunking. See scripts/resolve.py.
            "anchor_text": anchor_from(passage),
            "word_overlap": round(ratio, 2),
            "notes": notes,
        }
        append(GOLDEN, record)
        used.add(record["id"])
        written += 1
        print(f"  saved {record['id']}  ({len(existing) + written} total)")

    print(f"\n{written} added, {skipped} skipped.")
    if skipped > written:
        print("More skipped than kept — that is expected under fixed-size chunking,")
        print("and it is worth a line in your write-up.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label questions for the golden set.")
    parser.add_argument("--section", help="Only show chunks from this section")
    parser.add_argument("--type", default="direct_lookup", choices=TYPES)
    parser.add_argument("-n", "--count", type=int, default=25,
                        help="Do 25 a sitting. Labels degrade after that.")
    parser.add_argument("--min-len", type=int, default=400)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)

    if args.stats:
        show_stats()
        return 0

    async def _main() -> None:
        try:
            await label_loop(args.section, args.type, args.count, args.min_len)
        finally:
            await dispose()

    asyncio.run(_main())
    show_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())