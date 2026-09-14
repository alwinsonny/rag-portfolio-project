"""Cross-encoder reranking: a second, slower opinion on the shortlist.

PHASE 4.

THE DISTINCTION THAT MATTERS:

  BI-ENCODER (your embedding model)
      Reads the question and the passage SEPARATELY, turning each into 384
      numbers, then compares the numbers. The separation is what makes fast
      search possible — every passage is embedded once, in advance, and search
      is a distance calculation. It is also the limitation: the model never sees
      question and passage together, so it cannot notice that a passage answers
      this particular question rather than merely being on-topic.

  CROSS-ENCODER (this module)
      Reads question and passage TOGETHER, in one forward pass, and outputs a
      relevance score. Far more accurate, because attention runs across both at
      once. Far too slow to run over 46,000 passages — every query would need
      46,000 forward passes.

So you use both, in order: retrieve widely with the bi-encoder, rerank narrowly
with the cross-encoder. Retrieve 50, rerank to 8.

WHY YOUR NUMBERS SAY THIS WILL PAY:

    recall@10   0.664     the right passage reaches the top ten
    recall@50   0.757     ...but it is in the top fifty

That 9-point gap is passages being FOUND BUT BURIED. Reranking adds no new
candidates — it only reorders the ones you already have — so that gap is
exactly and only what it can close. Expect recall@10 to climb toward 0.757 and
MRR to move more than recall does.

WHAT IT CANNOT DO:
The seven questions that find nothing in fifty candidates stay lost. A passage
that never entered the pool cannot be rescued by reordering it.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

from drug_label_rag.retrieval.dense import Hit
from drug_label_rag.settings import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder


@functools.lru_cache(maxsize=2)
def get_reranker(name: str | None = None) -> CrossEncoder:
    """Load and cache the cross-encoder.

    Imported inside the function so that importing this module stays cheap and
    fast unit tests never pull a model off disk. On Apple Silicon it picks up
    MPS automatically.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name or settings.reranker_model)


def rerank(
    query: str,
    hits: list[Hit],
    *,
    top_n: int | None = None,
    batch_size: int = 32,
) -> tuple[list[Hit], float]:
    """Score every candidate against the query, return the best top_n.

    Returns (reranked_hits, elapsed_ms). The timing is returned rather than
    logged because reranking buys accuracy with milliseconds, and you must
    report both sides of that trade — a latency figure you did not measure is
    not a trade-off, it is a hope.

    The scores replace the fusion scores, so the abstention gate downstream
    reads a single consistent scale. That matters: RRF scores are around 0.03
    and cross-encoder scores are unbounded logits, and thresholding on the wrong
    one silently does nothing.
    """
    if not hits:
        return [], 0.0

    limit = top_n or settings.rerank_top_n
    model = get_reranker()

    # One batched forward pass over all candidates. Scoring pairs one at a time
    # is several times slower for no benefit.
    pairs = [(query, hit.content) for hit in hits]
    start = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elapsed = (time.perf_counter() - start) * 1000

    scored = sorted(
        (
            Hit(
                chunk_id=hit.chunk_id,
                score=float(score),
                content=hit.content,
                section=hit.section,
                set_id=hit.set_id,
                brand_name=hit.brand_name,
                generic_name=hit.generic_name,
            )
            for hit, score in zip(hits, scores, strict=True)
        ),
        key=lambda h: h.score,
        reverse=True,
    )
    return scored[:limit], elapsed


if __name__ == "__main__":
    import argparse
    import asyncio

    from drug_label_rag.db.session import dispose, session_scope
    from drug_label_rag.retrieval.pipeline import retrieve

    parser = argparse.ArgumentParser(description="Compare retrieval before and after reranking.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5, help="Rows to display")
    parser.add_argument("--candidates", type=int, default=50)
    args = parser.parse_args()

    async def _main() -> None:
        question = " ".join(args.query)
        async with session_scope() as session:
            result = await retrieve(session, question, k=args.candidates)

        before = result.hits[: args.k]
        after, ms = rerank(question, result.hits, top_n=args.k)

        print(f"\n{len(result.hits)} candidates, reranked in {ms:.0f} ms\n")
        print("BEFORE (fusion order)")
        for rank, hit in enumerate(before, 1):
            print(f"  [{rank}] {hit.score:.5f}  {hit.citation}")
        print("\nAFTER (cross-encoder order)")
        moved = {h.chunk_id for h in after} - {h.chunk_id for h in before}
        for rank, hit in enumerate(after, 1):
            mark = " NEW" if hit.chunk_id in moved else "    "
            print(f"  [{rank}]{mark} {hit.score:+.3f}  {hit.citation}")
            print(f"        {' '.join(hit.content.split())[:150]}")
        await dispose()

    asyncio.run(_main())