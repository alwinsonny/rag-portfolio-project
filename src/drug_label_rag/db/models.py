"""Database schema: documents and chunks.

PHASE 1. Two tables. A document is one drug label; a chunk is a retrievable
piece of one.

DESIGN NOTE — DENORMALISATION IS DELIBERATE.
Chunks carry brand_name, generic_name and section even though those could be
joined from documents. At this corpus size the duplication costs nothing and it
makes retrieval a single-table query with metadata filters applied in the same
WHERE clause as the vector search. Normalise this and every retrieval becomes a
join, for no benefit.

DESIGN NOTE — WHY NO ALEMBIC YET.
The schema below is created with create_all in Phase 1. Alembic arrives in
Phase 3, when structure-aware chunking changes the chunk table for real and you
need a migration you can roll forward and back. Introducing migrations before
the first schema change is ceremony; introducing them after the second is too
late.
"""

from __future__ import annotations

from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from drug_label_rag.settings import settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One drug label. Keyed on set_id, which is stable across label revisions."""

    __tablename__ = "documents"

    set_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    brand_name: Mapped[str | None] = mapped_column(String(512))
    generic_name: Mapped[str | None] = mapped_column(String(512))
    substance_name: Mapped[str | None] = mapped_column(String(512))
    manufacturer: Mapped[str | None] = mapped_column(String(512))
    route: Mapped[str | None] = mapped_column(String(128))
    product_type: Mapped[str | None] = mapped_column(String(128))
    # Label version date. Phase 5's freshness policy compares these to find
    # superseded labels for the same molecule.
    effective_time: Mapped[date | None] = mapped_column(Date)
    # Keep the original record. Costs disk, saves you re-downloading when you
    # discover in Phase 3 that you want a field you did not extract.
    raw: Mapped[dict] = mapped_column(JSONB)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class Chunk(Base):
    """A retrievable passage."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    set_id: Mapped[str] = mapped_column(
        ForeignKey("documents.set_id", ondelete="CASCADE"), index=True
    )
    # The SPL section this came from. In Phase 1 (fixed chunking) this is
    # "all"; in Phase 3 it becomes the real section name and starts earning
    # its place as a filter and as enrichment text.
    section: Mapped[str] = mapped_column(String(64), index=True)
    position: Mapped[int] = mapped_column(Integer)

    # The original text, shown to the user.
    content: Mapped[str] = mapped_column(Text)
    # What actually got embedded. In Phase 1 identical to content; in Phase 3,
    # content prefixed with drug name and section title. Storing both means the
    # user sees clean text while retrieval searches the enriched version.
    embedded_text: Mapped[str] = mapped_column(Text)

    # Idempotency: re-running ingestion must not duplicate anything.
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))

    # Denormalised for filtering without a join.
    brand_name: Mapped[str | None] = mapped_column(String(512))
    generic_name: Mapped[str | None] = mapped_column(String(512))

    # Generated full-text column for BM25 in Phase 3. Added now so the Phase 3
    # change is a query, not a migration.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        # HNSW: better recall than IVFFlat at higher build cost and memory.
        # At a few thousand chunks either works — this is chosen deliberately
        # for recall, and the trade-off is worth an ADR.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id} {self.generic_name} {self.section} pos={self.position}>"


class IngestRun(Base):
    """One row per ingestion, so a measurement can be traced to a corpus state.

    Your recall@10 is only reproducible against a specific corpus snapshot and a
    specific chunking configuration. Recording both here means an evaluation
    result six weeks old is still interpretable.
    """

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[date] = mapped_column(Date, server_default=func.current_date())
    export_date: Mapped[str | None] = mapped_column(String(32))
    chunk_strategy: Mapped[str] = mapped_column(String(32))
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    enriched: Mapped[bool] = mapped_column(default=False)
    embedding_model: Mapped[str] = mapped_column(String(256))
    documents: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)