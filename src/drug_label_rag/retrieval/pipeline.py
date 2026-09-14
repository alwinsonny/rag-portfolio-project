"""The retrieval pipeline: run the arms, fuse the results.

PHASE 3. One entry point that every caller uses — the evaluation harness, the
CLI, and the Phase 5 service. Adding a stage (reranking, in Phase 4) means
changing this file and nothing else.

WHY THE ARMS RUN CONCURRENTLY:
Dense search and lexical search touch different indexes and do not depend on
each other, so running them at the same time makes hybrid retrieval cost about
the same wall-clock time as either arm alone. Running them one after the other
doubles your latency for no benefit — and latency is a Phase 5 acceptance
criterion, not an afterthought.

WHY EACH ARM GETS ITS OWN SESSION:
A SQLAlchemy session wraps a SINGLE database connection and one transaction. It
is not safe for concurrent use: two coroutines sharing one session will collide
while it is provisioning its connection, and SQLAlchemy raises

    InvalidRequestError: This session is provisioning a new connection;
    concurrent operations are not permitted

So concurrency has to come from the connection POOL, not from one session. Each
arm opens its own session, which checks out its own connection, runs its query,
and returns the connection. The caller's session is used only when a single arm
is requested, where no concurrency is involved.

This is the same distinction as threads sharing one file handle versus each
opening its own — and it is a genuinely common mistake when moving synchronous
database code to async.

WHY RETRIEVE 50 AND NOT 10:
Phase 4's cross-encoder can only reorder what it is given. Retrieving narrowly
here starves it: a passage that never enters the candidate pool cannot be
rescued by any amount of reranking. Retrieve widely, rerank narrowly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from drug_label_rag.db.session import session_scope
from drug_label_rag.retrieval.dense import Hit, dense_search
from drug_label_rag.retrieval.fusion import contribution_report, reciprocal_rank_fusion
from drug_label_rag.retrieval.lexical import lexical_search
from drug_label_rag.retrieval.rerank import rerank
from drug_label_rag.settings import settings


@dataclass(slots=True)
class RetrievalResult:
    hits: list[Hit]
    retrievers: list[str]
    # How many of the fused top 10 each arm supplied. If one arm never
    # contributes, it is not earning its latency — say so rather than leaving
    # it in because "hybrid" sounds better.
    contributions: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranked: bool = False
    # The candidate pool before reranking. Kept so the evaluation harness can
    # measure recall@50 on what retrieval actually found, while recall@10 is
    # measured on what reranking chose to surface.
    candidates: list[Hit] = field(default_factory=list)


async def retrieve(
    session: AsyncSession,
    query: str,
    *,
    use_dense: bool = True,
    use_lexical: bool = True,
    use_rerank: bool = False,
    k: int | None = None,
    top_n: int | None = None,
    generic_name: str | None = None,
    section: str | None = None,
    set_id: str | None = None,
) -> RetrievalResult:
    """Retrieve candidates for one query.

    The two boolean flags exist so an ablation is a parameter, not a code
    change: you can measure dense alone, lexical alone, and both fused, against
    the identical golden set, in three commands.
    """
    if not (use_dense or use_lexical):
        raise ValueError("Enable at least one retriever")

    filters = {"generic_name": generic_name, "section": section, "set_id": set_id}
    names: list[str] = []
    if use_dense:
        names.append("dense")
    if use_lexical:
        names.append("lexical")

    start = time.perf_counter()

    if len(names) == 1:
        # One arm: no concurrency, so reuse the caller's session directly.
        search = dense_search if names[0] == "dense" else lexical_search
        limit = settings.dense_candidates if names[0] == "dense" else settings.lexical_candidates
        rankings = [await search(session, query, k=k or limit, **filters)]
    else:
        # Two arms: each needs its OWN session, or they collide on the shared
        # connection. See the module docstring.
        async def run_dense() -> list[Hit]:
            async with session_scope() as own:
                return await dense_search(
                    own, query, k=k or settings.dense_candidates, **filters
                )

        async def run_lexical() -> list[Hit]:
            async with session_scope() as own:
                return await lexical_search(
                    own, query, k=k or settings.lexical_candidates, **filters
                )

        async with asyncio.TaskGroup() as group:
            dense_task = group.create_task(run_dense())
            lexical_task = group.create_task(run_lexical())
        rankings = [dense_task.result(), lexical_task.result()]

    elapsed = (time.perf_counter() - start) * 1000

    # A single arm needs no fusion — fusing one list only rewrites its scores.
    if len(rankings) == 1:
        hits = rankings[0]
    else:
        hits = reciprocal_rank_fusion(rankings, k=settings.rrf_k)

    candidates = hits
    rerank_ms = 0.0
    if use_rerank and hits:
        # Rerank the whole candidate pool, then keep the best few. Reranking a
        # shortlist of the shortlist would waste the accuracy you are paying
        # milliseconds for.
        hits, rerank_ms = rerank(query, hits, top_n=top_n)

    return RetrievalResult(
        hits=hits,
        candidates=candidates,
        reranked=use_rerank,
        retrievers=names,
        contributions=contribution_report(rankings, names, candidates, top_n=10)
        if len(rankings) > 1
        else {names[0]: min(10, len(candidates))},
        timings_ms={
            "retrieval_ms": round(elapsed, 1),
            "rerank_ms": round(rerank_ms, 1),
            "total_ms": round(elapsed + rerank_ms, 1),
        },
    )


if __name__ == "__main__":
    import argparse

    from drug_label_rag.db.session import dispose, session_scope

    parser = argparse.ArgumentParser(description="Try the retrieval pipeline.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-lexical", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    async def _main() -> None:
        async with session_scope() as session:
            result = await retrieve(
                session,
                " ".join(args.query),
                use_dense=not args.no_dense,
                use_lexical=not args.no_lexical,
                use_rerank=args.rerank,
            )
        t = result.timings_ms
        print(f"retrievers: {', '.join(result.retrievers)}"
              f"{' + rerank' if result.reranked else ''}   "
              f"retrieval {t['retrieval_ms']} ms"
              + (f", rerank {t['rerank_ms']} ms" if result.reranked else ""))
        print(f"top-10 contributions: {result.contributions}\n")
        for rank, hit in enumerate(result.hits[: args.k], 1):
            print(f"[{rank}] {hit.score:.5f}  {hit.citation}")
            print(f"    {' '.join(hit.content.split())[:200]}\n")
        await dispose()

    asyncio.run(_main())