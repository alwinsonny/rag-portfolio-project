"""Tests for the generation loop.

Every model call is mocked with respx, so the whole suite runs offline, free
and deterministically. The loop logic — gate, verify, repair once, abstain — is
what matters here, not the model's prose.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from drug_label_rag.answer.generate import (
    ABSTAIN_MESSAGE, Answer, Generator, answer_question, build_prompt,
)
from drug_label_rag.retrieval.dense import Hit

API = "https://api.anthropic.com/v1/messages"


def hit(cid: int, content: str, score: float = 5.0) -> Hit:
    return Hit(chunk_id=cid, score=score, content=content, section="warnings",
               set_id=f"set-{cid:04d}", brand_name=None, generic_name="LEVOFLOXACIN")


def reply(text: str) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 900, "output_tokens": 80},
    }


@pytest.fixture
def gen() -> Generator:
    return Generator(api_key="test", model="test-model")


PASSAGES = [
    hit(1, "Levofloxacin is present in human milk following oral administration."),
    hit(2, "A decision should be made whether to discontinue nursing or the drug."),
]


# ================= the gate runs before any model call =================

@respx.mock
async def test_low_score_abstains_without_calling_the_model(gen: Generator) -> None:
    """Refusing early is both safer and cheaper — no paid call for a result
    that was always going to be discarded."""
    route = respx.post(API).mock(return_value=httpx.Response(200, json=reply("x")))
    answer = await answer_question("anything", [hit(1, "text", score=-4.0)], gen,
                                   abstain_below=1.421)
    assert answer.abstained
    assert answer.abstain_reason == "below_threshold"
    assert answer.text == ABSTAIN_MESSAGE
    assert route.call_count == 0


async def test_no_results_abstains(gen: Generator) -> None:
    answer = await answer_question("anything", [], gen)
    assert answer.abstained
    assert answer.abstain_reason == "no_results"


@respx.mock
async def test_score_above_threshold_proceeds(gen: Generator) -> None:
    respx.post(API).mock(return_value=httpx.Response(
        200, json=reply("Levofloxacin is present in human milk [1].")))
    answer = await answer_question("in milk?", PASSAGES, gen, abstain_below=1.421)
    assert not answer.abstained
    assert answer.verified


# ===================== the verification loop =====================

@respx.mock
async def test_a_clean_answer_returns_on_the_first_attempt(gen: Generator) -> None:
    route = respx.post(API).mock(return_value=httpx.Response(
        200, json=reply("Levofloxacin is present in human milk [1]. A decision "
                        "should be made whether to discontinue nursing [2].")))
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.verified and answer.attempts == 1
    assert route.call_count == 1
    assert answer.verification.cited_markers == {1, 2}


@respx.mock
async def test_a_fabricated_citation_triggers_exactly_one_repair(gen: Generator) -> None:
    """The headline case: the model cites [7] when two passages were supplied."""
    route = respx.post(API).mock(side_effect=[
        httpx.Response(200, json=reply("Monitor renal function quarterly [7].")),
        httpx.Response(200, json=reply("Levofloxacin is present in human milk [1].")),
    ])
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.verified and answer.attempts == 2
    assert route.call_count == 2


@respx.mock
async def test_the_repair_message_names_the_offending_sentence(gen: Generator) -> None:
    import json as _json
    route = respx.post(API).mock(side_effect=[
        httpx.Response(200, json=reply("Most patients tolerate the 750 mg dose well.")),
        httpx.Response(200, json=reply("Levofloxacin is present in human milk [1].")),
    ])
    await answer_question("in milk?", PASSAGES, gen)
    second = _json.loads(route.calls[1].request.content)
    repair = second["messages"][-1]["content"]
    assert "750 mg" in repair


@respx.mock
async def test_two_failures_abstain_rather_than_trying_again(gen: Generator) -> None:
    """A second failure is systematic, not bad luck. Burning a third attempt
    hides the signal and costs three times as much."""
    route = respx.post(API).mock(return_value=httpx.Response(
        200, json=reply("Renal adjustment is required below 50 mL per minute.")))
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.abstained
    assert answer.abstain_reason == "verification_failed"
    assert answer.attempts == 2
    assert route.call_count == 2


@respx.mock
async def test_an_honest_refusal_from_the_model_is_accepted(gen: Generator) -> None:
    """Abstention must not be rejected for having no sources to cite."""
    respx.post(API).mock(return_value=httpx.Response(
        200, json=reply("The labelling does not cover that question.")))
    answer = await answer_question("cost?", PASSAGES, gen)
    assert answer.verified and not answer.abstained


# ===================== failures and accounting =====================

@respx.mock
async def test_a_permanent_api_error_abstains_rather_than_raising(gen: Generator) -> None:
    respx.post(API).mock(return_value=httpx.Response(400, text="bad request"))
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.abstained and answer.abstain_reason == "generation_failed"


@respx.mock
async def test_a_transient_error_is_retried(gen: Generator) -> None:
    route = respx.post(API).mock(side_effect=[
        httpx.Response(529, text="overloaded"),
        httpx.Response(200, json=reply("Present in human milk [1].")),
    ])
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.verified
    assert route.call_count == 2


@respx.mock
async def test_tokens_accumulate_across_repair_attempts(gen: Generator) -> None:
    respx.post(API).mock(side_effect=[
        # Must be a FACTUAL sentence to be a violation: a number, a unit or a
        # clinical verb. "Some text here." carries no claim and passes.
        httpx.Response(200, json=reply("The recommended dose is 500 mg daily.")),
        httpx.Response(200, json=reply("Present in human milk [1].")),
    ])
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert answer.input_tokens == 1800 and answer.output_tokens == 160


@respx.mock
async def test_citations_only_include_passages_actually_cited(gen: Generator) -> None:
    """The evidence panel shows what the answer used, not everything retrieved."""
    respx.post(API).mock(return_value=httpx.Response(
        200, json=reply("Levofloxacin is present in human milk [1].")))
    answer = await answer_question("in milk?", PASSAGES, gen)
    assert [c["marker"] for c in answer.citations] == [1]
    assert answer.citations[0]["set_id"] == "set-0001"


def test_the_prompt_carries_the_numbered_passages() -> None:
    from drug_label_rag.answer.assemble import assemble
    prompt = build_prompt("in milk?", assemble(PASSAGES))
    assert "[1]" in prompt and "[2]" in prompt
    assert "in milk?" in prompt
    assert "human milk" in prompt