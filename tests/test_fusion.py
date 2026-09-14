"""Tests for reciprocal rank fusion, worked on paper first.

Fusion bugs are silent: the system still returns results, they are just worse.
Every number below was computed by hand before the code ran.
"""
from __future__ import annotations

import pytest

from drug_label_rag.retrieval.dense import Hit
from drug_label_rag.retrieval.fusion import contribution_report, reciprocal_rank_fusion


def hits(*ids: int) -> list[Hit]:
    return [
        Hit(
            chunk_id=i,
            score=1.0,
            content="",
            section="",
            set_id="",
            brand_name=None,
            generic_name=None,
        )
        for i in ids
    ]



def test_single_list_preserves_order() -> None:
    out = reciprocal_rank_fusion([hits(7, 8, 9)])
    assert [h.chunk_id for h in out] == [7, 8, 9]


def test_score_is_one_over_k_plus_rank() -> None:
    """Rank 1 with k=60 -> 1/61 = 0.016393"""
    out = reciprocal_rank_fusion([hits(7)], k=60)
    assert out[0].score == pytest.approx(1 / 61)


def test_appearing_in_both_lists_beats_appearing_once() -> None:
    """chunk 8: 1/62 + 1/62 = 0.032258
       chunk 7: 1/61          = 0.016393
    So 8 wins despite never being first in either list."""
    dense = hits(7, 8)
    lexical = hits(9, 8)
    out = reciprocal_rank_fusion([dense, lexical], k=60)
    assert out[0].chunk_id == 8
    assert out[0].score == pytest.approx(2 / 62)


def test_deep_agreement_beats_shallow_disagreement() -> None:
    """The property that makes RRF useful: consensus outranks one confident hit.
    chunk 5: 1/62 + 1/62 = 0.032258   (rank 2 in both)
    chunk 1: 1/61 + 0    = 0.016393   (rank 1 in one only)
    """
    out = reciprocal_rank_fusion([hits(1, 5), hits(9, 5)], k=60)
    assert [h.chunk_id for h in out][:1] == [5]


def test_k_controls_how_much_a_top_rank_is_worth() -> None:
    """One confident hit at rank 1, versus weak agreement at rank 5 in both.

        chunk 1:  1/(k+1)
        chunk 5:  2/(k+5)

    k=1   ->  0.500 vs 0.333   the single strong hit wins
    k=60  ->  0.016 vs 0.031   agreement wins

    So k is not a magic constant: it sets how much you trust one retriever's
    confidence against two retrievers' consensus. The usual default of 60
    leans towards consensus, which is why hybrid retrieval helps at all.
    """
    lists = [hits(1, 2, 3, 4, 5), hits(9, 8, 7, 6, 5)]
    assert reciprocal_rank_fusion(lists, k=1)[0].chunk_id == 1
    assert reciprocal_rank_fusion(lists, k=60)[0].chunk_id == 5


def test_scores_are_incomparable_across_retrievers_and_that_is_fine() -> None:
    """Dense scores ~0.7, lexical ~0.05. RRF never reads them."""
    dense = [Hit(chunk_id=7, score=0.79), Hit(chunk_id=8, score=0.78)]
    lexical = [Hit(chunk_id=8, score=0.04), Hit(chunk_id=7, score=0.02)]
    out = reciprocal_rank_fusion([dense, lexical], k=60)
    assert {h.chunk_id for h in out} == {7, 8}
    # 7: 1/61 + 1/62 ; 8: 1/62 + 1/61 — identical, order is stable not arbitrary
    assert out[0].score == pytest.approx(out[1].score)


def test_weights_favour_one_retriever() -> None:
    out = reciprocal_rank_fusion([hits(7), hits(8)], k=60, weights=[3.0, 1.0])
    assert out[0].chunk_id == 7
    assert out[0].score == pytest.approx(3 / 61)


def test_mismatched_weights_raise() -> None:
    with pytest.raises(ValueError, match="one entry per ranking"):
        reciprocal_rank_fusion([hits(7), hits(8)], weights=[1.0])


def test_limit_truncates() -> None:
    out = reciprocal_rank_fusion([hits(1, 2, 3, 4, 5)], limit=3)
    assert len(out) == 3


def test_deduplicates_across_lists() -> None:
    out = reciprocal_rank_fusion([hits(7, 8), hits(8, 7)])
    assert len(out) == 2


def test_empty_lists_are_handled() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    assert [h.chunk_id for h in reciprocal_rank_fusion([hits(7), []])] == [7]


def test_contribution_report_counts_each_arm() -> None:
    dense, lexical = hits(1, 2, 3), hits(3, 4, 5)
    fused = reciprocal_rank_fusion([dense, lexical])
    report = contribution_report([dense, lexical], ["dense", "lexical"], fused, top_n=10)
    assert report == {"dense": 3, "lexical": 3}