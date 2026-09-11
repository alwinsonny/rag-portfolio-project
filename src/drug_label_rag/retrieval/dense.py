"""Dense retrieval: cosine similarity over pgvector.

PHASE 1. This is the whole of the naive baseline's retrieval, and it is
deliberately about thirty lines. Everything that makes retrieval good arrives in
Phases 3 and 4; this exists to be beaten.

ON THE DISTANCE OPERATOR: `<=>` is cosine distance in pgvector — smaller is
closer, 0 means identical. Similarity is 1 - distance, which is what this module
returns because a score where bigger-is-better is far easier to reason about at
a threshold (Phase 4) and to fuse (Phase 3).

The vectors were normalised at ingestion, and the index is built with
vector_cosine_ops. All three have to agree; if they do not, retrieval still
returns results, they are just quietly wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from drug_label_rag.db.models import Chunk
from drug_label_rag.ingest.embed import embed_query
from drug_label_rag.settings import settings


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved passage. The common currency of every retrieval stage.

    dense, lexical, fusion and reranking all produce and consume these, so a
    stage can be swapped or removed without touching its neighbours — which is
    exactly what an ablation requires.
    """

    chunk_id: int
    score: float
    content: str
    section: str
    set_id: str
    brand_name: str | None
    generic_name: str | None

    @property
    def citation(self) -> str:
        """How this passage is identified in an answer.

        Resolves to a set_id, not a drug name: the corpus contains several
        labels for the same molecule with different text, so a name alone does
        not identify a source.
        """
        name = self.generic_name or self.brand_name or "unknown"
        return f"{name} [{self.set_id[:8]}] — {self.section}"


def _apply_filters(
    stmt: Select,
    *,
    generic_name: str | None = None,
    section: str | None = None,
    set_id: str | None = None,
) -> Select:
    """Metadata filters, applied in the same query as the vector search.

    This is the concrete argument for keeping vectors in Postgres: filtering
    and similarity happen in one statement against one index-backed table,
    rather than filtering in a second system and reconciling the results.
    """
    if generic_name:
        stmt = stmt.where(Chunk.generic_name.ilike(f"%{generic_name}%"))
    if section:
        stmt = stmt.where(Chunk.section == section)
    if set_id:
        stmt = stmt.where(Chunk.set_id == set_id)
    return stmt


async def dense_search(
    session: AsyncSession,
    query: str,
    *,
    k: int | None = None,
    generic_name: str | None = None,
    section: str | None = None,
    set_id: str | None = None,
) -> list[Hit]:
    """Embed the query and return the k nearest chunks."""
    vector = embed_query(query)
    distance = Chunk.embedding.cosine_distance(vector)

    stmt = select(
        Chunk.id,
        Chunk.content,
        Chunk.section,
        Chunk.set_id,
        Chunk.brand_name,
        Chunk.generic_name,
        distance.label("distance"),
    )
    stmt = _apply_filters(
        stmt, generic_name=generic_name, section=section, set_id=set_id
    )
    stmt = stmt.order_by(distance).limit(k or settings.dense_candidates)

    rows = (await session.execute(stmt)).all()
    return [
        Hit(
            chunk_id=row.id,
            score=1.0 - float(row.distance),  # similarity: bigger is better
            content=row.content,
            section=row.section,
            set_id=row.set_id,
            brand_name=row.brand_name,
            generic_name=row.generic_name,
        )
        for row in rows
    ]


if __name__ == "__main__":
    import argparse
    import asyncio

    from drug_label_rag.db.session import dispose, session_scope

    parser = argparse.ArgumentParser(description="Try a dense retrieval query.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--section")
    parser.add_argument("--generic")
    args = parser.parse_args()

    async def _main() -> None:
        async with session_scope() as session:
            hits = await dense_search(
                session,
                " ".join(args.query),
                k=args.k,
                section=args.section,
                generic_name=args.generic,
            )
        for rank, hit in enumerate(hits, 1):
            print(f"\n[{rank}] {hit.score:.4f}  {hit.citation}")
            print(f"    {' '.join(hit.content.split())[:220]}")
        await dispose()

    asyncio.run(_main())