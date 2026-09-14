"""The abstention threshold sweep.

PHASE 4.

    python -m drug_label_rag.eval.abstention
    python -m drug_label_rag.eval.abstention --margin      # threshold on the gap instead
    python -m drug_label_rag.eval.abstention --plot        # write a PNG of the curve

THE DECISION THIS MAKES:
Below what score should the system refuse to answer?

Set it too low and it answers questions the corpus cannot support, fabricating
confidently from irrelevant passages. Set it too high and it refuses questions
it could have answered. There is no correct value — only a defensible one, and
the defence comes from the domain.

HOW IT WORKS:
Run both question sets through the full pipeline and record the top score for
each. That gives two distributions:

    answerable    scores where the system SHOULD answer
    unanswerable  scores where it SHOULD refuse

Then, for every candidate threshold, count the errors on both sides. Plot them
against each other and choose an operating point.

WHY YOU MUST REPORT BOTH NUMBERS:
"90% correct abstention" alone is meaningless — a system that refuses everything
scores 100%. Correct abstention and false abstention, always together, with the
chosen operating point marked. Reporting one side of a trade-off means you have
not understood that it is a trade-off.

A NOTE ON CROSS-ENCODER SCORES:
They are logits, not probabilities. Negative does not mean "irrelevant" — the
whole scale is shifted, and the useful boundary is wherever the two
distributions separate, not zero. Thresholding at zero because it looks like a
natural midpoint is a mistake this sweep exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from drug_label_rag.db.session import dispose, session_scope
from drug_label_rag.eval.retrieval import load_golden
from drug_label_rag.retrieval.pipeline import retrieve
from drug_label_rag.settings import settings

UNANSWERABLE = Path("data/unanswerable.jsonl")
RESULTS = Path("data/results")


@dataclass(frozen=True, slots=True)
class Scored:
    id: str
    question: str
    top_score: float
    # Gap between the best and second-best candidate. When several passages are
    # weakly plausible and none is right, this is small — which is exactly the
    # shape of a class-confusion failure (asking about dapagliflozin and getting
    # empagliflozin). Thresholding on margin catches cases absolute score misses.
    margin: float
    top_hit: str | None


async def score_set(questions: list[tuple[str, str]], k: int = 20) -> list[Scored]:
    """Run questions through the pipeline and record the top score for each."""
    out: list[Scored] = []
    async with session_scope() as session:
        for qid, text in questions:
            result = await retrieve(session, text, k=k, use_rerank=True, top_n=2)
            hits = result.hits
            top = float(hits[0].score) if hits else -99.0
            second = float(hits[1].score) if len(hits) > 1 else top
            out.append(
                Scored(
                    id=qid,
                    question=text,
                    top_score=top,
                    margin=top - second,
                    top_hit=hits[0].citation if hits else None,
                )
            )
    return out


def sweep(
    answerable: list[Scored],
    unanswerable: list[Scored],
    *,
    use_margin: bool = False,
    steps: int = 40,
) -> list[dict[str, float]]:
    """For every candidate threshold, count the errors on both sides."""
    key = (lambda s: s.margin) if use_margin else (lambda s: s.top_score)
    values = [key(s) for s in answerable + unanswerable]
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0

    rows: list[dict[str, float]] = []
    for i in range(steps + 1):
        threshold = lo + span * i / steps
        # Refuse when the score is BELOW the threshold.
        wrongly_refused = sum(1 for s in answerable if key(s) < threshold)
        wrongly_answered = sum(1 for s in unanswerable if key(s) >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "false_abstention": wrongly_refused / len(answerable),
                "false_answer": wrongly_answered / len(unanswerable),
                "correct_abstention": 1 - wrongly_answered / len(unanswerable),
                "n_wrongly_refused": wrongly_refused,
                "n_wrongly_answered": wrongly_answered,
            }
        )
    return rows


def recommend(rows: list[dict[str, float]], max_false_abstention: float) -> dict[str, float]:
    """Pick the threshold that refuses the most junk within a false-abstention budget.

    The budget is the policy decision. In a clinical context you spend it
    generously: a confident wrong statement about a contraindication is far
    worse than "I cannot answer that from the labelling."
    """
    eligible = [r for r in rows if r["false_abstention"] <= max_false_abstention]
    if not eligible:
        return rows[0]
    return max(eligible, key=lambda r: r["correct_abstention"])


def describe(name: str, scored: list[Scored]) -> None:
    values = sorted(s.top_score for s in scored)
    margins = sorted(s.margin for s in scored)
    print(f"\n{name}  ({len(scored)} questions)")
    print(f"  top score   min {values[0]:+.3f}   median {statistics.median(values):+.3f}"
          f"   max {values[-1]:+.3f}")
    print(f"  margin      min {margins[0]:+.3f}   median {statistics.median(margins):+.3f}"
          f"   max {margins[-1]:+.3f}")


def overlap_report(answerable: list[Scored], unanswerable: list[Scored]) -> None:
    """How badly the two distributions interleave.

    Perfect separation means one threshold sorts everything. Overlap means some
    errors are unavoidable at ANY threshold, and the honest thing is to say so
    rather than tuning until one number looks good.
    """
    ans_min = min(s.top_score for s in answerable)
    una_max = max(s.top_score for s in unanswerable)

    print("\nOVERLAP")
    if una_max < ans_min:
        print(f"  None. Every unanswerable question scores below every answerable one")
        print(f"  ({una_max:+.3f} < {ans_min:+.3f}). A single threshold sorts them "
              f"perfectly.")
        return

    trapped_ans = [s for s in answerable if s.top_score <= una_max]
    trapped_una = [s for s in unanswerable if s.top_score >= ans_min]
    print(f"  The distributions overlap between {ans_min:+.3f} and {una_max:+.3f}.")
    print(f"  {len(trapped_ans)} answerable and {len(trapped_una)} unanswerable "
          f"questions fall in that band,")
    print("  so no single threshold on absolute score separates them. Some errors")
    print("  are unavoidable — try --margin, or accept and report the trade-off.")

    if trapped_una:
        print("\n  Hardest unanswerable cases (highest scoring):")
        for s in sorted(trapped_una, key=lambda s: -s.top_score)[:5]:
            print(f"    {s.top_score:+.3f}  {s.question[:62]}")
            print(f"             retrieved: {s.top_hit}")


def print_sweep(rows: list[dict[str, float]], chosen: dict[str, float]) -> None:
    print(f"\n{'threshold':>10}  {'refuses junk':>13}  {'refuses good':>13}")
    print("-" * 42)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["n_wrongly_answered"]), int(row["n_wrongly_refused"]))
        if key in seen:
            continue  # collapse identical rows; only transitions are interesting
        seen.add(key)
        mark = " <-- chosen" if row is chosen else ""
        print(f"{row['threshold']:>10.3f}  "
              f"{row['correct_abstention']:>12.0%}  "
              f"{row['false_abstention']:>12.0%}{mark}")


async def main_async(args: argparse.Namespace) -> None:
    golden = load_golden()
    unanswerable_rows = [
        json.loads(line)
        for line in UNANSWERABLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("scoring answerable set ...")
    answerable = await score_set([(q.id, q.question) for q in golden])
    print("scoring unanswerable set ...")
    unanswerable = await score_set([(r["id"], r["question"]) for r in unanswerable_rows])

    describe("ANSWERABLE", answerable)
    describe("UNANSWERABLE", unanswerable)
    overlap_report(answerable, unanswerable)

    rows = sweep(answerable, unanswerable, use_margin=args.margin)
    chosen = recommend(rows, args.max_false_abstention)

    metric = "margin" if args.margin else "top score"
    print(f"\nSWEEP on {metric}   (budget: refuse at most "
          f"{args.max_false_abstention:.0%} of answerable questions)")
    print_sweep(rows, chosen)

    print(f"\nRECOMMENDED THRESHOLD  {chosen['threshold']:+.3f}")
    print(f"  correctly refuses  {chosen['correct_abstention']:.0%} of "
          f"{len(unanswerable)} unanswerable questions")
    print(f"  wrongly refuses    {chosen['false_abstention']:.0%} of "
          f"{len(answerable)} answerable questions")
    print(f"\n  Set it in .env:   ABSTAIN_BELOW_SCORE={chosen['threshold']:.3f}")
    print("  Then write an ADR recording WHY this operating point, not another.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"abstention-{'margin' if args.margin else 'score'}.json"
    out.write_text(
        json.dumps(
            {
                "metric": metric,
                "reranker": settings.reranker_model,
                "n_answerable": len(answerable),
                "n_unanswerable": len(unanswerable),
                "chosen": chosen,
                "curve": rows,
                "answerable": [asdict(s) for s in answerable],
                "unanswerable": [asdict(s) for s in unanswerable],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out}")

    if args.plot:
        write_plot(rows, chosen, metric)


def write_plot(rows: list[dict[str, Any]], chosen: dict[str, float], metric: str) -> None:
    """The single figure that belongs above the fold in your README."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")
        return

    x = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["false_answer"] for r in rows], label="answers unanswerable questions")
    ax.plot(x, [r["false_abstention"] for r in rows], label="refuses answerable questions")
    ax.axvline(chosen["threshold"], linestyle="--", linewidth=1, color="grey")
    ax.annotate(
        f"chosen {chosen['threshold']:+.2f}",
        xy=(chosen["threshold"], 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel(f"abstention threshold ({metric})")
    ax.set_ylabel("error rate")
    ax.set_title("Abstention trade-off")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = RESULTS / f"abstention-{'margin' if metric == 'margin' else 'score'}.png"
    fig.savefig(path, dpi=150)
    print(f"plot written to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the abstention threshold.")
    parser.add_argument("--margin", action="store_true",
                        help="Threshold on the top-to-second gap rather than the top score")
    parser.add_argument("--max-false-abstention", type=float, default=0.10,
                        help="How many answerable questions you are willing to refuse")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    async def _main() -> None:
        try:
            await main_async(args)
        finally:
            await dispose()

    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())