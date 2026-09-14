"""Lexical retrieval: Postgres full-text search.

PHASE 3. The other half of hybrid retrieval.

WHY THIS EXISTS, measured on your own corpus:

    named_entity   recall@10  0.375     <- dense retrieval failing
    paraphrase     recall@10  0.714
    direct_lookup  recall@10  0.700

Dense embeddings compress meaning, and in doing so they blur exactly the tokens
that matter most in drug labelling: "O'Brien Fleming", "Child-Pugh score 7-9",
"4 mg by 15-minute infusion", NDC codes. A question naming one of those needs
the retriever to match the STRING, not the vibe. That is what lexical search
does, and it is the only reason to add a second retriever at all.

THE MISTAKE THIS MODULE ORIGINALLY MADE — worth understanding:

The first version used `websearch_to_tsquery`, chosen because it parses ordinary
prose forgivingly and never raises on punctuation. But it combines terms with
AND. A natural-language question becomes:

    "estimated baseline background risks of major birth defects"
      ->  estimat & baselin & background & risk & major & birth & defect

Every one of those must appear in the SAME chunk. Almost no passage satisfies
seven simultaneous terms, so lexical search returned essentially nothing:
recall@50 of 0.086, and 0.000 on named-entity questions — the very type it
should win.

The fix is to combine with OR and let the RANKING decide. A chunk containing
six of the seven terms, tightly clustered, outranks one containing two. That is
exactly what ts_rank_cd measures, and it is how BM25-style retrieval is meant to
behave: every term contributes, none is mandatory.

WHY POSTGRES FULL-TEXT AND NOT THE rank-bm25 LIBRARY:
The tsvector column and its GIN index already exist — they were added to the
schema in Phase 1 precisely so this phase would be a query rather than a
migration. Keeping both retrievers in one store means one connection, one
transaction, and metadata filters that apply identically to both arms.

The trade-off, which you should be able to state: Postgres `ts_rank_cd` is not
literally BM25. It is a coverage-density ranking without BM25's document-length
normalisation or its saturating term-frequency curve. For passages that are all
roughly one size — yours are, by construction — the practical difference is
small. If you want true BM25, rank-bm25 over an in-memory index is the
alternative, at the cost of holding the corpus in RAM and rebuilding it on every
process start.
"""

from __future__ import annotations

import re

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from drug_label_rag.db.models import Chunk
from drug_label_rag.retrieval.dense import Hit, _apply_filters
from drug_label_rag.settings import settings


# Keep letters, digits and internal hyphens. Everything else — parentheses,
# apostrophes, commas — is punctuation that to_tsquery would choke on.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def build_or_query(query: str, max_terms: int = 24) -> str:
    """Turn a natural-language question into an OR-combined tsquery string.

    Postgres normalises and drops stopwords itself, so "the" and "of" cost
    nothing here. The cap exists because an extremely long question produces a
    query that matches half the corpus and ranks badly.

    Returns an empty string when nothing usable survives, which the caller
    treats as "no lexical matches" rather than letting Postgres raise.
    """
    terms = _TOKEN.findall(query)
    # Deduplicate while preserving order: repeated terms do not help ts_rank_cd
    # and make the query longer for no benefit.
    seen: dict[str, None] = {}
    for term in terms:
        if len(term) > 1:
            seen.setdefault(term.lower())
    return " | ".join(list(seen)[:max_terms])


async def lexical_search(
    session: AsyncSession,
    query: str,
    *,
    k: int | None = None,
    generic_name: str | None = None,
    section: str | None = None,
    set_id: str | None = None,
) -> list[Hit]:
    """Rank chunks by term overlap with the query.

    Terms are combined with OR, not AND. See the module docstring — ANDing them
    is what made the first version of this function return nothing at all.
    """
    or_terms = build_or_query(query)
    if not or_terms:
        return []
    tsquery = func.to_tsquery("english", or_terms)

    # ts_rank_cd weights by coverage density — how tightly the matched terms
    # cluster — which suits passage retrieval better than plain ts_rank.
    rank = func.ts_rank_cd(Chunk.tsv, tsquery)

    stmt = select(
        Chunk.id,
        Chunk.content,
        Chunk.section,
        Chunk.set_id,
        Chunk.brand_name,
        Chunk.generic_name,
        rank.label("rank"),
    ).where(Chunk.tsv.op("@@")(tsquery))

    stmt = _apply_filters(
        stmt, generic_name=generic_name, section=section, set_id=set_id
    )
    stmt = stmt.order_by(rank.desc()).limit(k or settings.lexical_candidates)

    rows = (await session.execute(stmt)).all()
    return [
        Hit(
            chunk_id=row.id,
            score=float(row.rank),
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

    parser = argparse.ArgumentParser(description="Try a lexical retrieval query.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    async def _main() -> None:
        async with session_scope() as session:
            hits = await lexical_search(session, " ".join(args.query), k=args.k)
        if not hits:
            print("No lexical matches. Every query term is absent from the corpus.")
        for rank, hit in enumerate(hits, 1):
            print(f"\n[{rank}] {hit.score:.4f}  {hit.citation}")
            print(f"    {' '.join(hit.content.split())[:220]}")
        await dispose()

    asyncio.run(_main())