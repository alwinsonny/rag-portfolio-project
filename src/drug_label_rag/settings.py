"""Typed configuration, validated at startup.

PHASE 1. Carried over from Project 01, extended with every knob you will tune.

THE POINT OF PUTTING TUNING PARAMETERS HERE: an ablation becomes a config change,
not a code edit. In Phase 3 you will sweep chunk_size, then the RRF constant,
then the candidate pool. If those values are scattered through the code you
cannot run a sweep, and you will not remember which run used which value.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- database ---
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/druglabels",
        description="Async driver. Alembic uses the sync equivalent.",
    )
    db_echo: bool = False

    # --- corpus filtering (PHASE 1) ---
    max_documents: int = Field(default=400, ge=10)
    require_sections: list[str] = Field(
        default=[
            "indications_and_usage",
            "contraindications",
            "dosage_and_administration",
        ],
        description="A record is kept only if all of these are populated.",
    )
    prescription_only: bool = True
    max_labels_per_generic: int = Field(
        default=3,
        ge=1,
        description=(
            "Keeping 2-3 labels per molecule is deliberate: it creates the "
            "near-duplicate retrieval problem reranking exists to solve. "
            "Keeping forty is just noise."
        ),
    )

    # --- chunking (PHASE 1 naive, PHASE 3 structure-aware) ---
    chunk_strategy: str = Field(default="fixed", pattern="^(fixed|section)$")
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    enrich_with_parent_context: bool = Field(
        default=False,
        description="PHASE 3. Prepend drug name and section title before embedding. "
        "Measure this on its own — it is usually the cheapest win in the project.",
    )

    # --- embedding ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384, description="MUST match the model's output.")
    embedding_batch_size: int = Field(default=64, ge=1)

    # --- retrieval (PHASE 3-4) ---
    dense_candidates: int = Field(default=50, ge=1)
    lexical_candidates: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal rank fusion constant.")
    rerank_top_n: int = Field(default=8, ge=1)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- abstention (PHASE 4) ---
    abstain_below_score: float = Field(
        default=0.0,
        description="Set this from the threshold sweep, not by guessing.",
    )

    # --- generation (PHASE 5) ---
    model_api_key: str = ""
    model_name: str = ""
    context_token_budget: int = Field(default=6000, ge=500)


settings = Settings()