"""Reciprocal Rank Fusion: combining two ranked lists.

PHASE 3.

THE PROBLEM IT SOLVES:
Dense retrieval returns cosine similarities, roughly 0.4 to 0.9. Lexical
retrieval returns ts_rank_cd scores, which are unbounded and typically tiny —
0.05, 0.1. Adding or averaging those numbers is meaningless: whichever scale
happens to be larger silently dominates. Normalising them (min-max, z-score)
seems reasonable and is fragile, because the distributions shift from query to
query — a query with one strong lexical match and a query with twenty weak ones
normalise to completely different things.

THE FIX:
Ignore the scores entirely and use only RANK POSITION.

    RRF(d) = sum over retrievers of  1 / (k + rank(d))

A document at rank 1 in one list and rank 30 in another beats a document at
rank 5 in both. The constant k dampens the advantage of the very top positions,
so one retriever cannot dominate on a single confident hit.

This is why RRF is the default in practice: it needs no calibration, no
training, and no assumptions about score distributions. It is also about
fifteen lines of code, which is worth noticing.
"""

from __future__ import annotations

from collections.abc import Sequence

from drug_label_rag.retrieval.dense import Hit
from drug_label_rag.settings import settings


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    *,
    k: int | None = None,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[Hit]:
    """Fuse several ranked lists into one.

    `weights` lets you favour one retriever over another. Plain RRF weights
    both equally, which is the honest starting point — if you weight them,
    measure the difference and report it rather than tuning until the number
    looks good.

    The returned Hit carries the fused score, not the original one, so
    downstream stages (reranking, the abstention gate) see a single consistent
    ordering.
    """
    constant = k if k is not None else settings.rrf_k
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for position, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (
                constant + position
            )
            # Keep the first Hit seen for each chunk — the content and metadata
            # are identical whichever retriever found it.
            best.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if limit:
        ordered = ordered[:limit]

    return [
        Hit(
            chunk_id=chunk_id,
            score=score,
            content=best[chunk_id].content,
            section=best[chunk_id].section,
            set_id=best[chunk_id].set_id,
            brand_name=best[chunk_id].brand_name,
            generic_name=best[chunk_id].generic_name,
        )
        for chunk_id, score in ordered
    ]


def contribution_report(
    rankings: Sequence[Sequence[Hit]],
    names: Sequence[str],
    fused: Sequence[Hit],
    top_n: int = 10,
) -> dict[str, int]:
    """How many of the fused top N came from each retriever.

    Useful for the write-up: if the lexical arm contributes nothing to the top
    10 on any query, it is not earning its latency, and you should say so
    rather than leaving it in because hybrid sounds better.
    """
    top_ids = {hit.chunk_id for hit in fused[:top_n]}
    return {
        name: len({hit.chunk_id for hit in ranking} & top_ids)
        for name, ranking in zip(names, rankings, strict=True)
    }