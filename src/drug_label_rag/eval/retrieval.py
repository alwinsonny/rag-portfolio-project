"""The evaluation harness: run the golden set, score it, record the result.

PHASE 1 onward. This is the tool you will run dozens of times.

    python -m drug_label_rag.eval.retrieval
    python -m drug_label_rag.eval.retrieval --label "baseline: fixed chunks, dense only"
    python -m drug_label_rag.eval.retrieval --compare

EVERY RUN WRITES A TIMESTAMPED RESULTS FILE recording the FULL configuration
alongside the metrics — chunk strategy, size, overlap, enrichment, embedding
model, candidate count. By Phase 4 you will have run this thirty times and you
will not remember which settings produced which number. The ablation table in
your write-up is assembled from these files, not from memory.

The per-question-type breakdown is where the interesting findings live.
"recall@10 rose from 0.61 to 0.87" is a result; "driven almost entirely by
named-entity questions going from 0.38 to 0.79 once BM25 was added, while
paraphrase questions barely moved" is the answer of someone who ran the
experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from drug_label_rag.db.session import dispose, session_scope
from drug_label_rag.eval.metrics import evaluate, evaluate_by_group
from drug_label_rag.retrieval.dense import dense_search
from drug_label_rag.settings import settings

GOLDEN = Path("data/golden.jsonl")
RESULTS_DIR = Path("data/results")


class Relevant(BaseModel):
    chunk_id: int
    grade: int = Field(default=2, ge=1, le=3)


class GoldenQuestion(BaseModel):
    """One labelled question.

    `set_id` and `section` are stored ALONGSIDE chunk_id deliberately. Chunk IDs
    are reassigned on every re-ingest, and Phase 3 re-ingests constantly — these
    two fields are what let you re-resolve the labels afterwards instead of
    relabelling 150 questions from scratch.
    """

    id: str
    type: str = "direct_lookup"
    question: str
    relevant: list[Relevant]
    set_id: str | None = None
    section: str | None = None
    notes: str = ""


def load_golden(path: Path = GOLDEN) -> list[GoldenQuestion]:
    if not path.exists():
        raise SystemExit(
            f"No golden set at {path}. Write 20 questions by hand first "
            f"(scripts/label.py) — Phase 1 needs a provisional number, and "
            f"Phase 2 grows this to 150."
        )
    with path.open(encoding="utf-8") as fh:
        return [GoldenQuestion.model_validate_json(line) for line in fh if line.strip()]


def config_snapshot() -> dict[str, Any]:
    """Everything that could change a number. If it is not here, an old result
    file is uninterpretable."""
    return {
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "enriched": settings.enrich_with_parent_context,
        "embedding_model": settings.embedding_model,
        "dense_candidates": settings.dense_candidates,
        "retrievers": ["dense"],  # Phase 3 adds "bm25" and "rrf"
        "reranker": None,  # Phase 4 fills this in
    }


async def run_evaluation(
    questions: list[GoldenQuestion], k: int = 50
) -> tuple[list[tuple[str, list[int], dict[int, int]]], list[dict[str, Any]]]:
    """Retrieve for every question. Returns (runs_for_metrics, per_question_detail)."""
    runs: list[tuple[str, list[int], dict[int, int]]] = []
    detail: list[dict[str, Any]] = []

    async with session_scope() as session:
        for question in questions:
            hits = await dense_search(session, question.question, k=k)
            retrieved = [h.chunk_id for h in hits]
            judgements = {r.chunk_id: r.grade for r in question.relevant}
            runs.append((question.type, retrieved, judgements))

            found_at = next(
                (i for i, cid in enumerate(retrieved, 1) if cid in judgements), None
            )
            detail.append(
                {
                    "id": question.id,
                    "type": question.type,
                    "question": question.question,
                    "first_relevant_rank": found_at,
                    "top_hit": hits[0].citation if hits else None,
                }
            )
    return runs, detail


def write_results(
    label: str,
    metrics: dict[str, float],
    by_type: dict[str, dict[str, float]],
    detail: list[dict[str, Any]],
    n_questions: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "label": label,
                "timestamp": stamp,
                "questions": n_questions,
                "config": config_snapshot(),
                "metrics": metrics,
                "by_type": by_type,
                "per_question": detail,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def print_report(
    label: str, metrics: dict[str, float], by_type: dict[str, dict[str, float]], n: int
) -> None:
    print()
    print("=" * 68)
    print(f"{label}   ({n} questions)")
    print("=" * 68)
    cfg = config_snapshot()
    print(f"  {cfg['chunk_strategy']} chunks, size {cfg['chunk_size']}, "
          f"overlap {cfg['chunk_overlap']}, enriched={cfg['enriched']}")
    print(f"  retrievers: {', '.join(cfg['retrievers'])}")
    print()
    for k in (1, 5, 10, 20, 50):
        if f"recall@{k}" in metrics:
            print(f"  k={k:<3} recall {metrics[f'recall@{k}']:.3f}   "
                  f"mrr {metrics[f'mrr@{k}']:.3f}   ndcg {metrics[f'ndcg@{k}']:.3f}")
    print()
    print("  BY QUESTION TYPE (k=10)")
    for group, values in sorted(by_type.items()):
        marker = "  " if group != "ALL" else "* "
        print(f"  {marker}{group:<20} recall {values['recall@10']:.3f}   "
              f"ndcg {values['ndcg@10']:.3f}")
    print("=" * 68)


def compare() -> None:
    """Print every recorded run as one table. This is your ablation."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No results yet.")
    print(f"\n{'label':<44} {'r@10':>6} {'mrr':>6} {'ndcg':>6}  config")
    print("-" * 100)
    for path in files:
        data = json.loads(path.read_text())
        m, c = data["metrics"], data["config"]
        cfg = (f"{c['chunk_strategy']}/{c['chunk_size']}"
               f"{'/enriched' if c['enriched'] else ''} "
               f"[{'+'.join(c['retrievers'])}]")
        print(f"{data['label'][:44]:<44} {m.get('recall@10', 0):>6.3f} "
              f"{m.get('mrr@10', 0):>6.3f} {m.get('ndcg@10', 0):>6.3f}  {cfg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval.")
    parser.add_argument("--label", default="", help="Describe this configuration")
    parser.add_argument("--suite", default=str(GOLDEN))
    parser.add_argument("-k", type=int, default=50, help="Candidates to retrieve")
    parser.add_argument("--compare", action="store_true", help="Show all past runs")
    args = parser.parse_args(argv)

    if args.compare:
        compare()
        return 0

    questions = load_golden(Path(args.suite))
    label = args.label or (
        f"{settings.chunk_strategy} chunks, dense only"
        + (", enriched" if settings.enrich_with_parent_context else "")
    )

    async def _main() -> None:
        try:
            runs, detail = await run_evaluation(questions, k=args.k)
        finally:
            await dispose()

        metrics = evaluate([(r, j) for _, r, j in runs])
        by_type = evaluate_by_group(runs, k=10)
        print_report(label, metrics, by_type, len(questions))

        missed = [d for d in detail if d["first_relevant_rank"] is None]
        if missed:
            print(f"\n{len(missed)} questions found NOTHING relevant in {args.k} "
                  f"candidates — these are retrieval failures, not ranking failures:")
            for d in missed[:8]:
                print(f"  [{d['id']}] {d['question'][:70]}")

        path = write_results(label, metrics, by_type, detail, len(questions))
        print(f"\nwritten to {path}")

    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())