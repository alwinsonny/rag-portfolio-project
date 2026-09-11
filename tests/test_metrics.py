"""Tests for the retrieval metrics.

Every assertion here is a number worked out on paper first. This is the most
important test file in Project 02: if these are wrong, every measurement in the
ablation table is wrong, and you will not find out until someone checks your
work in an interview.
"""

from __future__ import annotations

import math

import pytest

from drug_label_rag.metrics import (
    dcg_at_k,
    evaluate,
    evaluate_by_group,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ======================================================================
# recall@k
# ======================================================================


def test_recall_finds_the_only_relevant_item() -> None:
    assert recall_at_k([7, 2, 9], relevant=[7], k=3) == 1.0


def test_recall_is_position_blind() -> None:
    """A hit at rank 1 and a hit at rank 10 score identically. That is the point."""
    early = recall_at_k([7, 1, 2, 3, 4, 5, 6, 8, 9, 10], [7], k=10)
    late = recall_at_k([1, 2, 3, 4, 5, 6, 8, 9, 10, 7], [7], k=10)
    assert early == late == 1.0


def test_recall_respects_the_cutoff() -> None:
    """Relevant item sits at rank 4, so it is outside k=3."""
    assert recall_at_k([1, 2, 3, 7], [7], k=3) == 0.0
    assert recall_at_k([1, 2, 3, 7], [7], k=4) == 1.0


def test_recall_with_multiple_relevant_items() -> None:
    """Two of three relevant items found -> 2/3."""
    assert recall_at_k([7, 1, 8, 2], relevant=[7, 8, 9], k=4) == pytest.approx(2 / 3)


def test_recall_with_no_relevant_items_is_zero_not_an_error() -> None:
    assert recall_at_k([1, 2, 3], relevant=[], k=3) == 0.0


# ======================================================================
# MRR
# ======================================================================


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        ([7, 1, 2, 3], 1.0),      # rank 1 -> 1/1
        ([1, 7, 2, 3], 0.5),      # rank 2 -> 1/2
        ([1, 2, 7, 3], 1 / 3),    # rank 3
        ([1, 2, 3, 7], 0.25),     # rank 4
        ([1, 2, 3, 4], 0.0),      # not found
    ],
)
def test_reciprocal_rank_is_one_over_rank(retrieved: list[int], expected: float) -> None:
    assert reciprocal_rank(retrieved, [7], k=4) == pytest.approx(expected)


def test_mrr_only_counts_the_first_relevant_hit() -> None:
    """Both 7 and 8 are relevant; only the earlier one matters."""
    assert reciprocal_rank([1, 7, 8, 2], relevant=[7, 8], k=4) == 0.5


def test_mrr_averages_across_queries() -> None:
    """(1.0 + 0.5 + 0.0) / 3 = 0.5 exactly."""
    runs = [
        ([7, 1, 2], [7]),
        ([1, 7, 2], [7]),
        ([1, 2, 3], [7]),
    ]
    assert mean_reciprocal_rank(runs, k=3) == pytest.approx(0.5)


# ======================================================================
# DCG / nDCG — worked on paper
# ======================================================================


def test_dcg_single_perfect_hit_at_rank_one() -> None:
    """grade 2 at rank 1: (2^2 - 1) / log2(2) = 3 / 1 = 3.0"""
    assert dcg_at_k([7], {7: 2}, k=10) == pytest.approx(3.0)


def test_dcg_discounts_by_position() -> None:
    """Same grade-2 item at rank 3: 3 / log2(4) = 3 / 2 = 1.5"""
    assert dcg_at_k([1, 2, 7], {7: 2}, k=10) == pytest.approx(1.5)


def test_dcg_sums_across_relevant_items() -> None:
    """grade 2 at rank 1 plus grade 1 at rank 2:
    3/log2(2) + 1/log2(3) = 3.0 + 0.63093 = 3.63093
    """
    expected = 3.0 + 1.0 / math.log2(3)
    assert dcg_at_k([7, 8], {7: 2, 8: 1}, k=10) == pytest.approx(expected)


def test_ndcg_is_one_for_the_ideal_ordering() -> None:
    assert ndcg_at_k([7, 8], {7: 2, 8: 1}, k=10) == pytest.approx(1.0)


def test_ndcg_penalises_a_swapped_ordering() -> None:
    """Grade 1 first, grade 2 second:
        actual = 1/log2(2) + 3/log2(3) = 1.0 + 1.89279 = 2.89279
        ideal  = 3/log2(2) + 1/log2(3) = 3.0 + 0.63093 = 3.63093
        nDCG   = 0.79671
    """
    actual = 1.0 + 3.0 / math.log2(3)
    ideal = 3.0 + 1.0 / math.log2(3)
    assert ndcg_at_k([8, 7], {7: 2, 8: 1}, k=10) == pytest.approx(actual / ideal)
    assert ndcg_at_k([8, 7], {7: 2, 8: 1}, k=10) == pytest.approx(0.7967, abs=1e-4)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k([1, 2, 3], {7: 2}, k=10) == 0.0


def test_ndcg_with_no_judgements_is_zero_not_a_divide_by_zero() -> None:
    """Guard against 0/0 silently scoring 1.0."""
    assert ndcg_at_k([1, 2, 3], {}, k=10) == 0.0


def test_ndcg_normalisation_makes_queries_comparable() -> None:
    """A query with three relevant passages, all retrieved in ideal order,
    scores 1.0 — the same as a query with one. Without normalisation the first
    would score three times higher purely for having more gain available.
    """
    one = ndcg_at_k([7], {7: 2}, k=10)
    three = ndcg_at_k([7, 8, 9], {7: 2, 8: 2, 9: 1}, k=10)
    assert one == pytest.approx(1.0)
    assert three == pytest.approx(1.0)


def test_ndcg_respects_the_cutoff() -> None:
    """The relevant item at rank 3 is invisible at k=2."""
    assert ndcg_at_k([1, 2, 7], {7: 2}, k=2) == 0.0
    assert ndcg_at_k([1, 2, 7], {7: 2}, k=3) > 0.0


# ======================================================================
# The behaviour that makes all three worth quoting
# ======================================================================


def test_reranking_moves_mrr_but_not_recall() -> None:
    """The signature of a reranking win: same candidates, better order.

    This is the exact pattern you should see in Phase 4, and being able to
    recognise it is why you quote all three metrics.
    """
    before = [1, 2, 7]   # relevant item buried at rank 3
    after = [7, 1, 2]    # same set, reordered
    judgements = {7: 2}

    assert recall_at_k(before, judgements.keys(), 10) == recall_at_k(
        after, judgements.keys(), 10
    )
    assert reciprocal_rank(after, judgements.keys(), 10) > reciprocal_rank(
        before, judgements.keys(), 10
    )
    assert ndcg_at_k(after, judgements, 10) > ndcg_at_k(before, judgements, 10)


def test_wider_retrieval_moves_recall_but_not_mrr() -> None:
    """The signature of a retrieval win: found more, still buried."""
    narrow = [1, 2, 3]
    wide = [1, 2, 3, 4, 5, 7]
    rel = [7]

    assert recall_at_k(wide, rel, 10) > recall_at_k(narrow, rel, 10)
    assert reciprocal_rank(wide, rel, 3) == reciprocal_rank(narrow, rel, 3) == 0.0


# ======================================================================
# Aggregation
# ======================================================================


def test_evaluate_produces_every_metric_at_every_k() -> None:
    runs = [([7, 1, 2], {7: 2}), ([1, 8, 2], {8: 1})]
    out = evaluate(runs, ks=(1, 5))
    assert set(out) == {"recall@1", "mrr@1", "ndcg@1", "recall@5", "mrr@5", "ndcg@5"}
    assert out["recall@1"] == pytest.approx(0.5)   # first query only
    assert out["recall@5"] == pytest.approx(1.0)


def test_evaluate_on_empty_input_returns_empty() -> None:
    assert evaluate([]) == {}


def test_group_breakdown_separates_question_types() -> None:
    runs = [
        ("named_entity", [7, 1, 2], {7: 2}),
        ("named_entity", [1, 2, 3], {9: 2}),
        ("paraphrase", [8, 1, 2], {8: 2}),
    ]
    out = evaluate_by_group(runs, k=10)
    assert out["named_entity"]["recall@10"] == pytest.approx(0.5)
    assert out["paraphrase"]["recall@10"] == pytest.approx(1.0)
    assert out["ALL"]["recall@10"] == pytest.approx(2 / 3)