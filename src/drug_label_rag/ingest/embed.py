"""Embedding.

PHASE 1.

TWO THINGS THAT MATTER HERE:

1. BATCH, ALWAYS. Embedding one chunk at a time turns a four-minute ingest into
   a two-hour one, and you will re-ingest repeatedly in Phase 3 while tuning
   chunking. This is the difference between iterating and waiting.

2. NORMALISE. The chunks table indexes with vector_cosine_ops, and cosine
   similarity assumes unit-length vectors. sentence-transformers can normalise
   for you; getting this wrong produces retrieval that is subtly bad rather
   than obviously broken, which is far harder to notice.

The model is loaded lazily and cached, so importing this module stays cheap and
your fast unit tests never pull a few hundred megabytes off disk.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

from drug_label_rag.settings import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


@functools.lru_cache(maxsize=2)
def get_model(name: str | None = None) -> SentenceTransformer:
    """Load and cache the embedding model.

    Imported inside the function so that `import embed` costs nothing. On Apple
    Silicon sentence-transformers picks up MPS automatically, which makes this
    several times faster than CPU.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name or settings.embedding_model)


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[list[float]]:
    """Embed a list of texts. Returns unit-length vectors.

    A progress bar is not decoration: a silent twenty-minute run is
    indistinguishable from a hang, which is a lesson from Project 01.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size or settings.embedding_batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # required for cosine distance
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query. Same model, same normalisation as ingestion.

    Using a different model — or forgetting to normalise — here than at
    ingestion time is a classic silent failure: retrieval still returns
    results, they are just wrong.
    """
    return embed_texts([text], show_progress=False)[0]


def check_dimension() -> int:
    """Verify the model's output width matches settings.embedding_dim.

    A mismatch fails at INSERT time with a confusing Postgres error about
    vector dimensions, usually after you have already spent ten minutes
    embedding. Call this before ingesting.
    """
    actual = int(get_model().get_sentence_embedding_dimension())
    if actual != settings.embedding_dim:
        raise ValueError(
            f"Model {settings.embedding_model!r} produces {actual}-dimensional "
            f"vectors but settings.embedding_dim is {settings.embedding_dim}. "
            f"Set EMBEDDING_DIM={actual} and recreate the schema — the column "
            f"width is fixed at table creation."
        )
    return actual


if __name__ == "__main__":
    dim = check_dimension()
    sample = embed_query("What is the maximum daily dose for renal impairment?")
    norm = sum(x * x for x in sample) ** 0.5
    print(f"model      {settings.embedding_model}")
    print(f"dimension  {dim}")
    print(f"norm       {norm:.6f}   (must be ~1.0 for cosine)")