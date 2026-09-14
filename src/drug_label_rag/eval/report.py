"""Assemble the evaluation report from everything you have measured.

PHASE 5. The report is the deliverable — not the code, not the final number.

    python -m drug_label_rag.eval.report            # print to stdout
    python -m drug_label_rag.eval.report --out REPORT.md

WHY THIS IS A SCRIPT AND NOT A DOCUMENT YOU WRITE BY HAND:
Every evaluation run wrote a timestamped JSON to data/results/ recording the
full configuration alongside the metrics. By now there are a dozen of them, and
nobody remembers which settings produced which number. Assembling the table
from the files rather than from memory is the only way it is trustworthy — and
it means re-running one configuration updates the report automatically.

WHAT THE REPORT MUST CONTAIN, and why each part:

  ABLATION TABLE        Which change earned which part of the gain. Anyone can
                        say "I built a RAG system"; a table showing the effect
                        of each intervention is a different claim.

  PER-TYPE BREAKDOWN    Where the interesting findings live. "recall@10 rose"
                        is a result; "driven by named-entity questions, while
                        paraphrase questions barely moved" is a finding.

  ABSTENTION CURVE      Correct abstention AND false abstention, together, with
                        the operating point marked. One without the other is
                        meaningless — a system that refuses everything scores
                        100% on the first.

  NEGATIVE RESULTS      Interventions that made things worse. These are more
                        credible than five that all worked, because they show
                        you were measuring rather than confirming.

  LIMITATIONS           Written plainly. A reader who finds a limitation you
                        did not mention discounts everything else.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS = Path("data/results")
GOLDEN = Path("data/golden.jsonl")
UNANSWERABLE = Path("data/unanswerable.jsonl")
MANIFEST = Path("data/raw/manifest.json")


def load_runs() -> list[dict[str, Any]]:
    """Every retrieval evaluation, oldest first."""
    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("abstention"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "metrics" in data:
            data["_file"] = path.name
            runs.append(data)
    return runs


def load_abstention() -> dict[str, Any] | None:
    path = RESULTS / "abstention-score.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def describe_config(config: dict[str, Any]) -> str:
    parts = [f"{config.get('chunk_strategy')} / {config.get('chunk_size')}"]
    if config.get("enriched"):
        parts.append("enriched")
    parts.append("+".join(config.get("retrievers", ["dense"])))
    if config.get("reranker"):
        parts.append("rerank")
    return ", ".join(parts)


def ablation_table(runs: list[dict[str, Any]]) -> str:
    """One row per configuration, with the change from the previous row."""
    lines = [
        "| Configuration | recall@10 | recall@50 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| {run.get('label') or describe_config(run['config'])} "
            f"| {m.get('recall@10', 0):.3f} | {m.get('recall@50', 0):.3f} "
            f"| {m.get('mrr@10', 0):.3f} | {m.get('ndcg@10', 0):.3f} |"
        )
    return "\n".join(lines)


def by_type_table(run: dict[str, Any]) -> str:
    by_type = run.get("by_type", {})
    if not by_type:
        return "_No per-type breakdown recorded._"
    lines = ["| Question type | recall@10 | nDCG@10 |", "|---|---|---|"]
    for name in sorted(by_type):
        values = by_type[name]
        label = "**ALL**" if name == "ALL" else name
        lines.append(
            f"| {label} | {values.get('recall@10', 0):.3f} "
            f"| {values.get('ndcg@10', 0):.3f} |"
        )
    return "\n".join(lines)


def comparison(runs: list[dict[str, Any]], a_key: str, b_key: str) -> str:
    """Per-type delta between two configurations, to show WHERE a change acted."""
    a = next((r for r in runs if a_key in (r.get("label") or "")), None)
    b = next((r for r in runs if b_key in (r.get("label") or "")), None)
    if not (a and b and a.get("by_type") and b.get("by_type")):
        return ""
    lines = [
        f"| Question type | {a['label'][:34]} | {b['label'][:34]} | change |",
        "|---|---|---|---|",
    ]
    for name in sorted(set(a["by_type"]) & set(b["by_type"])):
        before = a["by_type"][name].get("recall@10", 0)
        after = b["by_type"][name].get("recall@10", 0)
        arrow = "+" if after >= before else ""
        lines.append(
            f"| {name} | {before:.3f} | {after:.3f} | {arrow}{after - before:.3f} |"
        )
    return "\n".join(lines)


def abstention_section(data: dict[str, Any] | None) -> str:
    if not data:
        return "_Abstention sweep not run._"
    chosen = data["chosen"]
    lines = [
        f"Threshold metric: **{data['metric']}**  ",
        f"Reranker: `{data['reranker']}`  ",
        f"Sets: {data['n_answerable']} answerable, {data['n_unanswerable']} unanswerable",
        "",
        "| Threshold | Correctly refuses unanswerable | Wrongly refuses answerable |",
        "|---|---|---|",
    ]
    seen: set[tuple[int, int]] = set()
    for row in data["curve"]:
        key = (int(row["n_wrongly_answered"]), int(row["n_wrongly_refused"]))
        if key in seen:
            continue
        seen.add(key)
        mark = " **&larr; chosen**" if abs(row["threshold"] - chosen["threshold"]) < 1e-9 else ""
        lines.append(
            f"| {row['threshold']:+.3f} | {row['correct_abstention']:.0%} "
            f"| {row['false_abstention']:.0%}{mark} |"
        )
    lines += [
        "",
        f"**Operating point {chosen['threshold']:+.3f}** — correctly refuses "
        f"{chosen['correct_abstention']:.0%} of unanswerable questions while wrongly "
        f"refusing {chosen['false_abstention']:.0%} of answerable ones.",
        "",
        "Both numbers, always together. Correct abstention alone is meaningless: a "
        "system that refuses everything scores 100%.",
    ]
    return "\n".join(lines)


def corpus_section() -> str:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    golden = sum(1 for _ in GOLDEN.open()) if GOLDEN.exists() else 0
    unanswerable = sum(1 for _ in UNANSWERABLE.open()) if UNANSWERABLE.exists() else 0
    return "\n".join([
        f"- **Source**: openFDA drug labelling, snapshot "
        f"`{manifest.get('export_date', 'unknown')}`",
        f"- **Partitions used**: {manifest.get('partitions_downloaded', '?')} of "
        f"{manifest.get('partitions_available', '?')}",
        "- **Filter**: human prescription drugs with indications, contraindications "
        "and dosage populated; one label per molecule",
        f"- **Evaluation set**: {golden} answerable questions, {unanswerable} "
        f"unanswerable questions, all written by hand",
        "",
        "Data courtesy of the U.S. Food and Drug Administration. The FDA does not "
        "endorse this tool. Public domain.",
    ])


def build(runs: list[dict[str, Any]], abstention: dict[str, Any] | None) -> str:
    best = max(runs, key=lambda r: r["metrics"].get("recall@10", 0)) if runs else None
    first = runs[0] if runs else None

    headline = ""
    if best and first:
        headline = (
            f"recall@10 improved from **{first['metrics'].get('recall@10', 0):.3f}** "
            f"({describe_config(first['config'])}) to "
            f"**{best['metrics'].get('recall@10', 0):.3f}** "
            f"({describe_config(best['config'])}) across {len(runs)} measured "
            f"configurations."
        )

    return f"""# Drug Label Retrieval — Evaluation Report

_Generated {datetime.now(UTC).strftime('%Y-%m-%d')}_

A retrieval system over official FDA drug labelling. Answers medication
questions with a citation on every factual claim, and refuses when the labelling
does not support an answer.

**Scope.** An information retrieval tool over published labelling, for
healthcare professionals. It reports what the label says. It does not give
patient-specific advice, does not recommend treatment, does not diagnose, and is
not a medical device.

## Headline

{headline}

## Corpus and evaluation set

{corpus_section()}

## Ablation

Each row is one configuration, measured against the same question set. Changes
were made one at a time so that each row's contribution is attributable.

{ablation_table(runs)}

## Where the gains came from

{by_type_table(best) if best else ''}

{comparison(runs, 'dense only', 'hybrid')}

## Abstention

{abstention_section(abstention)}

## Citation integrity

Every factual sentence in a generated answer must carry a marker resolving to a
passage that was actually supplied. This is checked in code, not requested in
the prompt: the answer is parsed, markers are resolved against the supplied
passages, and any factual sentence without one rejects the whole answer. The
system regenerates once with specific feedback, then abstains.

**This verifies citation integrity, not accuracy.** A marker can resolve
correctly to a passage that does not in fact support the claim. Measuring that
requires human review, which has not been done — see Limitations.

## Limitations

- **Faithfulness has not been measured.** Citations are verified to resolve;
  whether each cited passage actually supports its claim has not been checked by
  hand. One observed case: an answer about breastfeeding cited a passage about
  sun exposure alongside a correct one. The marker resolved; the support did not
  exist.
- **The evaluation set is small.** Differences below roughly 0.10 on 35
  questions are not distinguishable from noise, and no confidence intervals are
  reported.
- **The unanswerable set covers one failure mode.** All questions are
  "drug not in corpus". Questions about a section a label lacks, or about
  information labelling never contains (cost, comparative efficacy), are likely
  harder and are not represented.
- **Several unanswerable questions are meta-questions about the dataset**
  rather than clinical questions, which probably inflates the measured
  separation between the two score distributions.
- **A subset of questions is multi-clause** and fails in every configuration
  tested. These measure question phrasing rather than retrieval quality and cap
  the achievable recall.
- **Tabular content is retrieved but not usable.** Passages consisting mostly of
  numeric tables occupy candidate slots and can never serve as a citation.
- **One partition of fourteen** was ingested, and one label per molecule
  retained. Results do not generalise to the full corpus without re-measurement.

## Reproducing

```bash
docker compose up -d                          # or a local Postgres with pgvector
python -m drug_label_rag.ingest.download --limit 1
python -m drug_label_rag.db.session
python -m drug_label_rag.ingest.run --reset
python -m drug_label_rag.eval.retrieval --rerank
python -m drug_label_rag.eval.abstention
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the evaluation report.")
    parser.add_argument("--out", help="Write to a file instead of stdout")
    args = parser.parse_args(argv)

    runs = load_runs()
    if not runs:
        raise SystemExit(f"No evaluation results in {RESULTS}/. Run the harness first.")

    report = build(runs, load_abstention())
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"written to {args.out}  ({len(runs)} configurations)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())