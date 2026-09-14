"""Grounded generation: the only module in this project that calls a model.

PHASE 5.

WHAT THE MODEL SEES:
The question, and the numbered passages you chose. Nothing else. It has no
access to the database, no knowledge of the corpus, and no ability to look
anything up. Six weeks of retrieval work exists to make sure the eight passages
it receives are the right ones.

THE LOOP:

    assemble  ->  generate  ->  verify  ->  ok?      -> return
                       ^                    reject   -> regenerate ONCE
                       |                                     |
                       +-------------------------------------+
                                                        still bad -> abstain

ONE REPAIR ATTEMPT, then abstain. The same discipline as Project 01's dispatch
layer: a second failure is not bad luck, it is a signal that something
systematic is wrong — the passages do not support an answer, or the prompt is
ambiguous. Burning three attempts hides that signal and costs three times as
much.

WHY THE ABSTENTION CHECK COMES FIRST:
The gate reads the top reranker score BEFORE any model call. If the corpus
cannot support an answer, there is no point generating one and then throwing it
away — that is a paid API call for a result you were always going to discard.
Refusing early is both safer and cheaper.

WHY GENERATION IS BUFFERED, NOT STREAMED:
You cannot verify an answer you have not finished generating. Streaming tokens
straight to the user means an uncited claim is on their screen before the
verifier sees it, and retracting it afterwards is worse than a short wait.
Phase 5's latency budget accommodates this deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from drug_label_rag.answer.assemble import AssembledContext, Source, assemble
from drug_label_rag.answer.verify import VerificationResult, repair_instruction, verify
from drug_label_rag.retrieval.dense import Hit
from drug_label_rag.settings import settings

log = structlog.get_logger(__name__)

API_VERSION = "2023-06-01"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

SYSTEM_PROMPT = """\
You answer questions about medicines using ONLY the numbered passages supplied \
to you. Those passages are extracts from official FDA drug labelling.

Rules, in order of importance:

1. Every factual sentence must end with a source marker: [1], [2], and so on. \
A sentence may cite more than one passage.
2. Use ONLY what the passages say. Do not add information from your own \
knowledge, however confident you are, and however obvious it seems.
3. If the passages do not answer the question, say so plainly in one sentence. \
Do not assemble a partial answer from loosely related material, and do not \
hedge — an honest refusal is more useful than a vague answer.
4. If the passages concern a DIFFERENT drug from the one asked about, that is \
not an answer. Say the labelling supplied does not cover that drug.
5. Write for a healthcare professional: plain, specific, no filler. Two to five \
sentences unless the question genuinely needs more.
6. Never give advice about a particular patient. Report what the labelling \
says; do not recommend a course of action.\
"""

ABSTAIN_MESSAGE = (
    "The supplied labelling does not contain enough information to answer that "
    "question confidently."
)


class GenerationError(RuntimeError):
    """The model call failed in a way retrying will not fix."""


class TransientGenerationError(RuntimeError):
    """The model call failed in a way that may succeed on retry."""


@dataclass(slots=True)
class Answer:
    """What a question returns. Always this shape, answered or abstained."""

    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    verified: bool = False
    verification: VerificationResult | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    top_score: float | None = None

    @property
    def citations(self) -> list[dict[str, Any]]:
        """The evidence panel: what the user needs to check the answer.

        A professional must be able to verify without leaving the interface,
        which means the passage text, the label it came from, and its date.
        """
        cited = (
            self.verification.cited_markers
            if self.verification
            else {s.marker for s in self.sources}
        )
        return [
            {
                "marker": s.marker,
                "drug": s.drug,
                "section": s.section.replace("_", " "),
                "set_id": s.set_id,
                "text": s.content,
            }
            for s in self.sources
            if s.marker in cited
        ]


def _abstain(question: str, reason: str, *, top_score: float | None = None,
             sources: list[Source] | None = None) -> Answer:
    return Answer(
        question=question,
        text=ABSTAIN_MESSAGE,
        sources=sources or [],
        abstained=True,
        abstain_reason=reason,
        top_score=top_score,
    )


class Generator:
    """Thin wrapper over the Messages API. Every model call goes through here."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or settings.model_api_key
        self.model = model or settings.model_name or "claude-sonnet-4-6"
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.max_tokens = max_tokens
        # Zero temperature: the faithfulness review samples 60 answers, and a
        # measurement you cannot reproduce is not a measurement.
        self.temperature = temperature
        self._client = httpx.AsyncClient(timeout=TIMEOUT)

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(TransientGenerationError),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )
    async def _complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                },
            )
        except httpx.TransportError as exc:
            raise TransientGenerationError(str(exc)) from exc

        if response.status_code in (408, 429, 500, 502, 503, 504, 529):
            raise TransientGenerationError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise GenerationError(f"HTTP {response.status_code}: {response.text[:300]}")

        body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        usage = body.get("usage", {})
        return text.strip(), usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def build_prompt(question: str, context: AssembledContext) -> str:
    return (
        f"Passages from official drug labelling:\n\n{context.prompt_block}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the passages above, with a [n] marker on every "
        f"factual sentence."
    )


async def answer_question(
    question: str,
    hits: list[Hit],
    generator: Generator,
    *,
    abstain_below: float | None = None,
) -> Answer:
    """Produce a verified answer, or abstain.

    `hits` come from the retrieval pipeline with reranking already applied, so
    hits[0].score is the cross-encoder score the abstention gate reads.
    """
    threshold = (
        abstain_below if abstain_below is not None else settings.abstain_below_score
    )
    top_score = float(hits[0].score) if hits else None

    # --- gate before spending anything ---------------------------------
    if not hits:
        return _abstain(question, "no_results")
    if top_score is not None and top_score < threshold:
        log.info("abstained", reason="below_threshold", score=top_score)
        return _abstain(question, "below_threshold", top_score=top_score)

    context = assemble(hits)
    if not context.sources:
        return _abstain(question, "no_context", top_score=top_score)

    messages = [{"role": "user", "content": build_prompt(question, context)}]
    result: VerificationResult | None = None
    text = ""
    tokens_in = tokens_out = 0

    # --- generate, verify, one repair ----------------------------------
    for attempt in (1, 2):
        try:
            text, used_in, used_out = await generator._complete(messages)
        except (GenerationError, TransientGenerationError) as exc:
            log.warning("generation_failed", error=str(exc))
            return _abstain(question, "generation_failed", top_score=top_score,
                            sources=context.sources)
        tokens_in += used_in
        tokens_out += used_out

        result = verify(text, context.sources)
        if result.ok:
            return Answer(
                question=question,
                text=text,
                sources=context.sources,
                verified=True,
                verification=result,
                attempts=attempt,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                top_score=top_score,
            )

        log.warning("verification_failed", attempt=attempt, summary=result.summary)
        if attempt == 2:
            break

        # Specific feedback. Naming the offending sentence works; asking the
        # model to "cite better" does not.
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": repair_instruction(result)},
        ]

    # --- two failures is systematic, not bad luck ----------------------
    answer = _abstain(question, "verification_failed", top_score=top_score,
                      sources=context.sources)
    answer.verification = result
    answer.attempts = 2
    answer.input_tokens = tokens_in
    answer.output_tokens = tokens_out
    return answer


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    from drug_label_rag.db.session import dispose, session_scope
    from drug_label_rag.retrieval.pipeline import retrieve

    parser = argparse.ArgumentParser(description="Answer one question end to end.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    async def _main() -> None:
        question = " ".join(args.question)
        generator = Generator()
        try:
            async with session_scope() as session:
                result = await retrieve(
                    session, question, k=args.candidates, use_rerank=True, top_n=8
                )
            answer = await answer_question(question, result.hits, generator)
        finally:
            await generator.aclose()
            await dispose()

        if args.json:
            print(json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "abstained": answer.abstained,
                    "reason": answer.abstain_reason,
                    "verified": answer.verified,
                    "attempts": answer.attempts,
                    "top_score": answer.top_score,
                    "citations": answer.citations,
                },
                indent=2,
            ))
            return

        print(f"\nQ: {answer.question}")
        print(f"   top score {answer.top_score:+.3f}   "
              f"attempts {answer.attempts}   "
              f"{answer.input_tokens} in / {answer.output_tokens} out\n")
        if answer.abstained:
            print(f"ABSTAINED ({answer.abstain_reason})")
            print(f"  {answer.text}")
            if answer.verification:
                print(f"  {answer.verification.summary}")
                for violation, detail in answer.verification.violations[:4]:
                    print(f"    {violation.value}: {detail}")
            return

        print(answer.text)
        print(f"\n[{answer.verification.summary}]" if answer.verification else "")
        print("\nSOURCES")
        for citation in answer.citations:
            print(f"  [{citation['marker']}] {citation['drug']} — "
                  f"{citation['section']} ({citation['set_id'][:8]})")
            print(f"      {' '.join(citation['text'].split())[:190]}")

    asyncio.run(_main())