"""Author the unanswerable question set.

    python scripts/unanswerable.py              # write questions
    python scripts/unanswerable.py --drugs      # what IS in the corpus
    python scripts/unanswerable.py --stats
    python scripts/unanswerable.py --verify     # re-check the whole set

WHAT THIS SET IS FOR:
Phase 4's abstention gate. The system must say "the labelling does not answer
that" instead of assembling a confident answer from irrelevant passages. Without
questions that genuinely have no answer, you cannot measure whether it does.

Almost no RAG project builds this. It is the most differentiating part of the
whole project, and it is also the easiest to do badly.

WHAT MAKES A BAD UNANSWERABLE QUESTION:
"What is the capital of France?" — obviously out of domain. Any threshold, however
badly tuned, refuses it. Your abstention rate looks excellent and means nothing.

WHAT MAKES A GOOD ONE:
A question that looks exactly like an answerable one and happens not to be. The
retrieval will return plausible, on-topic passages with respectable scores — and
the gate still has to refuse. That is a real test.

FIVE KINDS THAT WORK, roughly in order of usefulness:

  1. A DRUG NOT IN THE CORPUS
     "What is the usual starting dose of metformin?"  Perfectly reasonable,
     and if metformin is not among your 400 labels, unanswerable. Use --drugs
     to see what you have, then pick common drugs you do not.

  2. A SECTION THIS LABEL LACKS
     Only 56% of labels have warnings_and_cautions; 33% have boxed_warning.
     Ask about a boxed warning for a drug whose label has none.

  3. SOMETHING LABELLING NEVER COVERS
     Price, availability, insurance coverage, which manufacturer is preferred,
     how it compares to a competitor drug. All plausible clinical questions,
     none of them in an SPL document.

  4. PATIENT-SPECIFIC ADVICE
     "Should my 62-year-old patient on warfarin take this?" Correctly outside
     scope — the tool reports what the label says, it does not advise.

  5. BEYOND THE SNAPSHOT
     "Has this been approved for paediatric use since 2026?" Your corpus is a
     fixed snapshot and cannot answer anything about later changes.

HOW THIS TOOL VERIFIES:
It runs your real retrieval pipeline on each candidate and shows you the top
hits. If one of them actually answers the question, the question is answerable
and gets rejected. Assuming unanswerability without checking is how this set
quietly fills up with questions your corpus can in fact answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from drug_label_rag.db.models import Document
from drug_label_rag.db.session import dispose, session_scope
from drug_label_rag.retrieval.pipeline import retrieve

UNANSWERABLE = Path("data/unanswerable.jsonl")

REASONS = [
    "drug_not_in_corpus",
    "section_absent",
    "not_in_labelling",
    "patient_specific",
    "beyond_snapshot",
]


def load(path: Path = UNANSWERABLE) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append(record: dict) -> None:
    UNANSWERABLE.parent.mkdir(parents=True, exist_ok=True)
    with UNANSWERABLE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def list_drugs(limit: int) -> None:
    """What is actually in the corpus, so you can pick drugs that are not."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Document.generic_name)
                .where(Document.generic_name.isnot(None))
                .order_by(func.lower(Document.generic_name))
            )
        ).all()
    names = sorted({r[0].strip().lower() for r in rows if r[0]})
    print(f"{len(names)} distinct generic names in the corpus\n")
    for i in range(0, min(len(names), limit), 3):
        print("  " + "".join(f"{n[:34]:<36}" for n in names[i : i + 3]))
    if len(names) > limit:
        print(f"\n  ... and {len(names) - limit} more")
    print("\nPick common drugs that are NOT on this list. Those make the best")
    print("unanswerable questions: entirely plausible, and genuinely absent.")


def show_stats() -> None:
    rows = load()
    print(f"\n{len(rows)} unanswerable questions (target 40)")
    for reason, n in Counter(r.get("reason", "?") for r in rows).most_common():
        print(f"  {reason:<22} {n}")
    if rows:
        scores = [r.get("top_score", 0.0) for r in rows]
        scores.sort()
        print(f"\ntop-hit score across the set:")
        print(f"  min {scores[0]:.3f}   median {scores[len(scores)//2]:.3f}   "
              f"max {scores[-1]:.3f}")
        print("\nThe HIGH end is what matters. A question whose best hit scores 0.9")
        print("is a hard case — the gate has to refuse something that looks right.")


async def author(count: int, reason: str) -> None:
    existing = load()
    seen = {r["question"].lower() for r in existing}
    written = rejected = 0

    print("Type a question your corpus should NOT be able to answer.")
    print("The tool will retrieve and show you what comes back.")
    print("Blank line to skip, 'q' to stop.\n")

    async with session_scope() as session:
        while written < count:
            try:
                question = input("question> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() == "q":
                break
            if not question:
                continue
            if question.lower() in seen:
                print("  already in the set\n")
                continue

            result = await retrieve(session, question, k=20, use_rerank=True, top_n=3)
            hits = result.hits
            top_score = float(hits[0].score) if hits else 0.0

            print(f"\n  Retrieval returned (top score {top_score:+.3f}):")
            for i, hit in enumerate(hits[:3], 1):
                print(f"    [{i}] {hit.score:+.3f}  {hit.citation}")
                print(f"        {' '.join(hit.content.split())[:170]}")

            answer = input(
                "\n  Does ANY of those actually answer the question? [y/N]> "
            ).strip().lower()
            if answer == "y":
                rejected += 1
                print("  -> answerable. Rejected — this would have been a false "
                      "abstention case.\n")
                continue

            kind = input(f"  reason [{reason}]> ").strip() or reason
            if kind not in REASONS:
                print(f"  unknown reason, using {reason}")
                kind = reason

            record = {
                "id": f"u-{len(existing) + written + 1:03d}",
                "question": question,
                "reason": kind,
                # Recorded so you can tune the abstention threshold against the
                # HARD cases — the ones whose best hit scored highly.
                "top_score": round(top_score, 4),
                "top_hit": hits[0].citation if hits else None,
            }
            append(record)
            seen.add(question.lower())
            written += 1
            print(f"  saved {record['id']}  ({len(existing) + written} of 40)\n")

    print(f"\n{written} added, {rejected} rejected as actually answerable.")


async def verify_all() -> None:
    """Re-check the whole set against the current corpus.

    Worth running after any re-ingest: a question that was unanswerable against
    one chunking may be answerable against another.
    """
    rows = load()
    if not rows:
        raise SystemExit("No unanswerable questions yet.")

    suspicious: list[tuple[str, float, str]] = []
    async with session_scope() as session:
        for row in rows:
            result = await retrieve(
                session, row["question"], k=20, use_rerank=True, top_n=1
            )
            score = float(result.hits[0].score) if result.hits else 0.0
            row["top_score"] = round(score, 4)
            if score > 0.0:  # cross-encoder logits: positive means "relevant"
                suspicious.append((row["id"], score, row["question"]))

    print(f"\nchecked {len(rows)} questions")
    if suspicious:
        print(f"\n{len(suspicious)} scored positively — review these by hand. A high")
        print("score does not prove the question is answerable, but it is where a")
        print("mislabelled entry would hide:\n")
        for qid, score, question in sorted(suspicious, key=lambda x: -x[1])[:10]:
            print(f"  {qid}  {score:+.3f}  {question[:66]}")
    else:
        print("Nothing scored positively.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Author unanswerable questions.")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="Do about 10 a sitting; they get harder to invent")
    parser.add_argument("--reason", default="drug_not_in_corpus", choices=REASONS)
    parser.add_argument("--drugs", action="store_true", help="List corpus drugs")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    if args.stats:
        show_stats()
        return 0

    async def _main() -> None:
        try:
            if args.drugs:
                await list_drugs(args.limit)
            elif args.verify:
                await verify_all()
            else:
                await author(args.count, args.reason)
        finally:
            await dispose()

    asyncio.run(_main())
    if not (args.drugs or args.verify):
        show_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())