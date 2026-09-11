"""The ingestion pipeline: raw partition -> documents -> chunks -> embeddings.

PHASE 1.

    python -m drug_label_rag.ingest.run                 # ingest with current settings
    python -m drug_label_rag.ingest.run --reset         # drop and rebuild first
    python -m drug_label_rag.ingest.run --limit 50      # quick smoke run

IDEMPOTENCY IS THE POINT. Every chunk carries a content hash, and inserts skip
conflicts. Re-running this must never duplicate anything — and you WILL re-run
it, repeatedly, once Phase 3 starts changing the chunking strategy. Without the
hash, your third re-ingest silently triples the corpus and every metric moves
for reasons unrelated to your changes.

Each run writes an IngestRun row recording the corpus snapshot and the full
chunking configuration. A recall@10 figure is only interpretable against a known
corpus state; six weeks later that row is the only thing that tells you which.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm

from drug_label_rag.db.models import Chunk, Document, IngestRun
from drug_label_rag.db.session import dispose, init_db, session_scope
from drug_label_rag.ingest.chunk import chunk_document
from drug_label_rag.ingest.embed import check_dimension, embed_texts
from drug_label_rag.ingest.parse import ParsedDocument, select_corpus
from drug_label_rag.settings import settings

RAW_DIR = Path("data/raw")


def load_partition(path: Path) -> list[dict[str, Any]]:
    """Read one downloaded partition.

    Loaded whole rather than streamed: a partition is large but fits in memory
    comfortably, and the filter in select_corpus needs to see every record to
    apply the per-generic cap sensibly.
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            return json.load(fh)["results"]


def read_manifest() -> dict[str, Any]:
    path = RAW_DIR / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


async def store_documents(docs: list[ParsedDocument]) -> int:
    """Upsert documents. Existing rows are left alone."""
    if not docs:
        return 0
    rows = [
        {
            "set_id": d.set_id,
            "brand_name": d.brand_name,
            "generic_name": d.generic_name,
            "substance_name": d.substance_name,
            "manufacturer": d.manufacturer,
            "route": d.route,
            "product_type": d.product_type,
            "effective_time": d.effective_time,
            "raw": d.raw,
        }
        for d in docs
    ]
    async with session_scope() as session:
        stmt = insert(Document).values(rows).on_conflict_do_nothing(
            index_elements=["set_id"]
        )
        await session.execute(stmt)
    return len(rows)


async def store_chunks(rows: list[dict[str, Any]], batch: int = 500) -> int:
    """Insert chunks, skipping any whose content hash already exists.

    on_conflict_do_nothing against the unique content_hash is what makes the
    whole pipeline safe to re-run.
    """
    inserted = 0
    for start in range(0, len(rows), batch):
        window = rows[start : start + batch]
        async with session_scope() as session:
            stmt = insert(Chunk).values(window).on_conflict_do_nothing(
                index_elements=["content_hash"]
            )
            result = await session.execute(stmt)
            inserted += result.rowcount or 0
    return inserted


async def ingest(limit: int | None = None, reset: bool = False) -> None:
    # Fail fast on a dimension mismatch, before spending ten minutes embedding.
    dim = check_dimension()

    partitions = sorted(RAW_DIR.glob("*.json.zip"))
    if not partitions:
        raise SystemExit(f"No partitions in {RAW_DIR}. Run the downloader first.")

    await init_db(drop=reset)

    manifest = read_manifest()
    print(f"corpus snapshot   {manifest.get('export_date', 'unknown')}")
    print(f"strategy          {settings.chunk_strategy} "
          f"(size={settings.chunk_size}, overlap={settings.chunk_overlap}, "
          f"enrich={settings.enrich_with_parent_context})")
    print(f"embedding         {settings.embedding_model} ({dim}d)")
    print()

    print(f"reading {partitions[0].name} ...")
    records = load_partition(partitions[0])
    docs = select_corpus(
        records,
        require_sections=settings.require_sections,
        prescription_only=settings.prescription_only,
        max_labels_per_generic=settings.max_labels_per_generic,
        max_documents=limit or settings.max_documents,
    )
    print(f"selected {len(docs)} documents from {len(records)} records")

    stored = await store_documents(docs)
    print(f"stored {stored} documents")

    # Chunk everything before embedding: one big batched encode is far faster
    # than one encode per document.
    all_chunks = []
    for doc in tqdm(docs, desc="chunking"):
        all_chunks.extend(
            chunk_document(
                doc,
                strategy=settings.chunk_strategy,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                enrich=settings.enrich_with_parent_context,
            )
        )
    if not all_chunks:
        raise SystemExit("No chunks produced — check the corpus filter.")

    lengths = sorted(len(c.content) for c in all_chunks)
    print(f"{len(all_chunks)} chunks  "
          f"(median {lengths[len(lengths)//2]}, max {lengths[-1]} chars, "
          f"{len(all_chunks)/len(docs):.1f} per document)")

    vectors = embed_texts([c.embedded_text for c in all_chunks])

    rows = [
        {
            "set_id": c.set_id,
            "section": c.section,
            "position": c.position,
            "content": c.content,
            "embedded_text": c.embedded_text,
            "content_hash": c.content_hash,
            "embedding": vector,
            "brand_name": c.brand_name,
            "generic_name": c.generic_name,
        }
        for c, vector in zip(all_chunks, vectors, strict=True)
    ]
    inserted = await store_chunks(rows)
    print(f"inserted {inserted} chunks ({len(rows) - inserted} already present)")

    async with session_scope() as session:
        session.add(
            IngestRun(
                export_date=str(manifest.get("export_date")),
                chunk_strategy=settings.chunk_strategy,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                enriched=settings.enrich_with_parent_context,
                embedding_model=settings.embedding_model,
                documents=len(docs),
                chunks=len(rows),
            )
        )

    async with session_scope() as session:
        total_docs = len((await session.execute(select(Document.set_id))).all())
        total_chunks = len((await session.execute(select(Chunk.id))).all())
    print(f"\ndatabase now holds {total_docs} documents and {total_chunks} chunks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the openFDA corpus.")
    parser.add_argument("--limit", type=int, help="Cap documents (smoke runs)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the schema first. Required after any chunking "
        "change, because old chunks would otherwise linger alongside new ones.",
    )
    args = parser.parse_args(argv)
    asyncio.run(_run(args.limit, args.reset))
    return 0


async def _run(limit: int | None, reset: bool) -> None:
    try:
        await ingest(limit=limit, reset=reset)
    finally:
        await dispose()


if __name__ == "__main__":
    raise SystemExit(main())