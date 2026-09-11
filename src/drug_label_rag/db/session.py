"""Database engine, sessions, and one-shot initialisation.

PHASE 1.

WHY ASYNC EVERYWHERE: the Phase 5 service runs an async request path, and a
synchronous driver inside an async handler silently serialises the entire
service — one slow query blocks every other request. That bug does not appear
until load testing, by which point it is expensive to unpick. Start async.

WHY create_all AND NOT ALEMBIC YET: the schema below is the first version.
Alembic arrives in Phase 3, when structure-aware chunking changes the chunk
table for real and you need a migration you can roll forward and back.
Introducing migrations before the first schema change is ceremony.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from drug_label_rag.db.models import Base
from drug_label_rag.settings import settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """One engine per process. Creating one per request exhausts connections."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # survives Postgres restarts without a stale-conn error
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """A session with commit-on-success, rollback-on-error, always closed."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(drop: bool = False) -> None:
    """Create the vector extension and the tables.

    The extension MUST exist before create_all runs — the chunks table has a
    vector column and an HNSW index, and both fail without it. This ordering is
    the single most common first-run error on this project.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        if drop:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def dispose() -> None:
    """Close the pool. Call this on shutdown, and in tests."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Create the schema.")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop everything first. You will use this constantly in Phase 3 "
        "while tuning chunking — re-chunking means re-ingesting from scratch.",
    )
    args = parser.parse_args()

    async def _main() -> None:
        await init_db(drop=args.drop)
        print("schema ready" + (" (dropped and recreated)" if args.drop else ""))
        await dispose()

    asyncio.run(_main())