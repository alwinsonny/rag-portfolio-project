"""Retrieval metrics: recall@k, MRR@k, nDCG@k.

PHASE 1. Implement these before you need them, and unit test every one against
an example you worked on paper. A wrong metric implementation invalidates the
entire project and is embarrassingly easy to get subtly wrong — nDCG in
particular has several variants in the wild.

WHAT EACH ONE ANSWERS:

  recall@k   "did we find it at all?"      — ignores position entirely
  MRR@k      "how quickly did we find it?" — only the FIRST relevant result
  nDCG@k     "is the whole ordering good?" — graded relevance, position-discounted

Quote all three. A phase that lifts MRR while leaving recall flat is a reranking
win; a phase that lifts recall while leaving MRR flat means you are finding more
but still burying it. You cannot tell those apart from one number.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

# A judgement maps a chunk id to a graded relevance score.
#   2 = fully answers the question
#   1 = partially relevant
#   0 = not relevant (usually just absent from the mapping)
Judgements = Mapping[int, int]


def recall_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of relevant items that appear anywhere in the top k.

    Position-blind: a relevant hit at rank 1 and at rank k score identically.
    This is the metric that tells you whether the right passage was even in the
    candidate pool — if recall@50 is low, no amount of reranking will save you.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = len(relevant_set & set(retrieved[:k]))
    return found / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """1 / rank of the FIRST relevant result, or 0.0 if none in the top k.

    Ranks are 1-based: a hit at position 0 in the list is rank 1 and scores 1.0.
    Falls away steeply — rank 2 is 0.5, rank 4 is 0.25.
    """
    relevant_set = set(relevant)
    for index, item in enumerate(retrieved[:k], start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[int], Iterable[int]]], k: int
) -> float:
    """MRR across a set of queries. Queries with no hit contribute 0."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel, k) for r, rel in results) / len(results)


def dcg_at_k(retrieved: Sequence[int], judgements: Judgements, k: int) -> float:
    """Discounted cumulative gain.

    Uses the standard exponential gain formulation:

        DCG = sum over i of  (2^rel_i - 1) / log2(i + 1)

    where i is the 1-based rank. The exponential form rewards a highly relevant
    result far more than two partially relevant ones, which is what you want
    when grade 2 means "fully answers" and grade 1 means "partially relevant".

    The linear variant (rel_i / log2(i+1)) also exists and is not wrong — but
    pick one, document it, and never mix them across measurements.
    """
    total = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        gain = judgements.get(item, 0)
        if gain:
            total += (2**gain - 1) / math.log2(index + 1)
    return total


def ndcg_at_k(retrieved: Sequence[int], judgements: Judgements, k: int) -> float:
    """Normalised DCG: actual ordering divided by the best possible ordering.

    Normalising makes scores comparable across queries — a question with three
    relevant passages cannot score higher than one with a single relevant
    passage simply because there was more gain available.

    Returns 0.0 when nothing is relevant, so a query with no judgements does not
    silently score 1.0 by dividing zero by zero.
    """
    ideal_order = [
        item for item, _ in sorted(judgements.items(), key=lambda kv: kv[1], reverse=True)
    ]
    ideal = dcg_at_k(ideal_order, judgements, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, judgements, k) / ideal


def evaluate(
    runs: Sequence[tuple[Sequence[int], Judgements]],
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> dict[str, float]:
    """Aggregate every metric over a set of queries.

    `runs` is a list of (retrieved_chunk_ids, judgements) — one entry per
    question in your golden set.
    """
    if not runs:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(
            recall_at_k(r, j.keys(), k) for r, j in runs
        ) / len(runs)
        out[f"mrr@{k}"] = sum(
            reciprocal_rank(r, j.keys(), k) for r, j in runs
        ) / len(runs)
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, j, k) for r, j in runs) / len(runs)
    return out


def evaluate_by_group(
    runs: Sequence[tuple[str, Sequence[int], Judgements]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Metrics broken down by question type.

    This is where the interesting findings live. "recall@10 rose from 0.61 to
    0.87" is good; "driven almost entirely by named-entity questions going from
    0.38 to 0.79 once BM25 was added, while paraphrase questions barely moved"
    is the answer of someone who actually ran the experiment.
    """
    groups: dict[str, list[tuple[Sequence[int], Judgements]]] = {}
    for group, retrieved, judgements in runs:
        groups.setdefault(group, []).append((retrieved, judgements))

    result = {g: evaluate(rs, ks=(k,)) for g, rs in groups.items()}
    result["ALL"] = evaluate([(r, j) for _, r, j in runs], ks=(k,))
    return result