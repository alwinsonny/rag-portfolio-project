"""The HTTP service.

PHASE 5.

    uvicorn drug_label_rag.api:app --reload
    open http://localhost:8000/docs

THE ENDPOINT THAT MATTERS is POST /ask: retrieve, rerank, gate, assemble,
generate, verify — the whole pipeline in one call.

TWO DESIGN DECISIONS WORTH DEFENDING:

1. AN ABSTENTION RETURNS 200, NOT AN ERROR.
   Refusing to answer is a correct outcome, not a failure. It gets a 200 with
   `abstained: true` and a reason. A 4xx would tell callers something went
   wrong, when in fact the system did exactly what it was built to do. The
   X-Abstained header lets a caller route these without parsing the body.

2. THE RESPONSE IS BUFFERED, NOT STREAMED.
   You cannot verify an answer you have not finished generating. Streaming
   would put an uncited claim on the user's screen before the verifier sees it,
   and retracting it afterwards is worse than a two-second wait. The latency
   budget accommodates this deliberately.

WHAT EVERY ANSWER CARRIES:
The cited passages, in full, with the drug, the section and the label's set_id.
A professional must be able to check the answer without leaving the interface —
that is the whole point of a retrieval system over a chatbot.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from drug_label_rag.answer.generate import Generator, answer_question
from drug_label_rag.db.session import dispose, session_scope
from drug_label_rag.retrieval.pipeline import retrieve
from drug_label_rag.settings import settings

log = structlog.get_logger(__name__)

_generator: Generator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """One Generator for the process, not one per request.

    Each Generator holds an httpx client with its own connection pool. Creating
    one per request throws away connection reuse and exhausts sockets under
    load — the same reason the database engine is a module-level singleton.
    """
    global _generator
    _generator = Generator()
    yield
    await _generator.aclose()
    await dispose()


app = FastAPI(
    title="Drug Label Retrieval",
    version="0.1.0",
    description=(
        "Answers medication questions from official FDA drug labelling, with a "
        "citation on every factual claim, and refuses when the labelling does "
        "not support an answer.\n\n"
        "**Scope.** An information retrieval tool over published labelling, for "
        "healthcare professionals. It reports what the label says. It does not "
        "give patient-specific advice, does not recommend treatment, does not "
        "diagnose, and is not a medical device."
    ),
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=500,
        description="A question about a medicine, as a professional would ask it.",
        examples=["Can a breastfeeding mother take levofloxacin?"],
    )
    candidates: int = Field(
        default=30, ge=5, le=100,
        description="How many passages to retrieve before reranking. More is "
        "slower and gives the reranker more to work with.",
    )
    sources: int = Field(
        default=8, ge=1, le=12,
        description="How many reranked passages to send to the model.",
    )


class Citation(BaseModel):
    marker: int
    drug: str
    section: str
    set_id: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    # True only when every factual sentence carried a marker resolving to a
    # supplied passage. Note this verifies CITATION INTEGRITY, not accuracy:
    # a marker can resolve to a passage that does not actually support the
    # claim. Measuring that requires human review.
    citations_verified: bool
    citations: list[Citation]
    top_score: float | None
    attempts: int
    tokens: dict[str, int]
    timings_ms: dict[str, float]


class SearchHit(BaseModel):
    rank: int
    score: float
    drug: str
    section: str
    set_id: str
    text: str


@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    """Liveness, plus a shape check on what the pipeline depends on."""
    from sqlalchemy import func, select

    from drug_label_rag.db.models import Chunk, Document

    async with session_scope() as session:
        documents = (await session.execute(select(func.count(Document.set_id)))).scalar_one()
        chunks = (await session.execute(select(func.count(Chunk.id)))).scalar_one()

    return {
        "status": "ok",
        "documents": documents,
        "chunks": chunks,
        "chunking": f"{settings.chunk_strategy}/{settings.chunk_size}",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "abstain_below_score": settings.abstain_below_score,
        "generation_configured": bool(settings.model_api_key),
    }


@app.post("/search", response_model=list[SearchHit], tags=["retrieval"])
async def search(request: AskRequest) -> list[SearchHit]:
    """Retrieval only — no model call, no cost.

    Useful for debugging an answer you disagree with: it shows exactly what the
    generator was given.
    """
    async with session_scope() as session:
        result = await retrieve(
            session, request.question, k=request.candidates,
            use_rerank=True, top_n=request.sources,
        )
    return [
        SearchHit(
            rank=i,
            score=round(float(hit.score), 4),
            drug=hit.generic_name or hit.brand_name or "unknown",
            section=hit.section.replace("_", " "),
            set_id=hit.set_id,
            text=" ".join(hit.content.split()),
        )
        for i, hit in enumerate(result.hits, 1)
    ]


@app.post("/ask", response_model=AskResponse, tags=["answer"])
async def ask(request: AskRequest, response: Response) -> AskResponse:
    """Answer a question, or refuse.

    Always 200. An abstention is a correct outcome, not an error — the
    X-Abstained header lets callers route without parsing the body.
    """
    if _generator is None:  # pragma: no cover - lifespan always runs
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not ready.")

    started = time.perf_counter()
    async with session_scope() as session:
        result = await retrieve(
            session, request.question, k=request.candidates,
            use_rerank=True, top_n=request.sources,
        )
    retrieval_ms = (time.perf_counter() - started) * 1000

    gen_started = time.perf_counter()
    answer = await answer_question(request.question, result.hits, _generator)
    generation_ms = (time.perf_counter() - gen_started) * 1000

    if answer.abstained:
        response.headers["X-Abstained"] = answer.abstain_reason or "unknown"

    log.info(
        "answered",
        abstained=answer.abstained,
        reason=answer.abstain_reason,
        verified=answer.verified,
        attempts=answer.attempts,
        top_score=answer.top_score,
        total_ms=round(retrieval_ms + generation_ms),
    )

    return AskResponse(
        question=answer.question,
        answer=answer.text,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        citations_verified=answer.verified,
        citations=[Citation(**c) for c in answer.citations],
        top_score=round(answer.top_score, 4) if answer.top_score is not None else None,
        attempts=answer.attempts,
        tokens={"input": answer.input_tokens, "output": answer.output_tokens},
        timings_ms={
            "retrieval": round(retrieval_ms, 1),
            "generation": round(generation_ms, 1),
            "total": round(retrieval_ms + generation_ms, 1),
        },
    )