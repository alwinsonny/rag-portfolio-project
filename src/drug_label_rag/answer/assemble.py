"""Context assembly: turning retrieved passages into a prompt the model can use.

PHASE 5.

THREE JOBS, in order of how much they matter:

1. DEDUPLICATION.
   SPL labels repeat themselves. The HIGHLIGHTS block at the top of a label
   restates sentences from the sections below, so the same fact routinely
   appears in two or three retrieved passages. Feeding all of them to the model
   wastes budget AND makes the answer sound more certain than the evidence
   warrants — three copies of one source reads like three sources.

2. TOKEN BUDGET.
   A model can only read so much. When the passages exceed the budget, something
   is dropped, and YOU decide what rather than letting the prompt truncate
   arbitrarily. The priority order is documented below and is a real design
   decision: dropping the highest-scoring passage because it happened to be last
   in the list is a bug you will not notice without a test.

3. NUMBERING.
   Each passage gets a marker — [1], [2] — so the model can point at it and the
   verifier can check. The marker is the whole basis of the citation guarantee.

WHAT THIS MODULE DOES NOT DO:
It never talks to a model. Pure functions over Hit objects, fully unit testable,
no API key required. Generation is the next module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from drug_label_rag.retrieval.dense import Hit
from drug_label_rag.settings import settings

# Rough characters-per-token for English prose. Deliberately conservative: an
# underestimate wastes a little budget, an overestimate silently truncates the
# prompt and the model never sees the passage you thought you sent.
CHARS_PER_TOKEN = 3.6


@dataclass(frozen=True, slots=True)
class Source:
    """One numbered passage, as the model and the user will both see it."""

    marker: int
    chunk_id: int
    content: str
    section: str
    set_id: str
    drug: str
    score: float

    @property
    def label(self) -> str:
        """Human-readable attribution.

        Resolves to a set_id, not just a drug name: several labels exist per
        molecule with different text, so a name alone does not identify a
        source.
        """
        pretty = self.section.replace("_", " ")
        return f"{self.drug} — {pretty} [{self.set_id[:8]}]"


@dataclass(slots=True)
class AssembledContext:
    sources: list[Source]
    prompt_block: str
    dropped_for_budget: int
    dropped_as_duplicate: int
    estimated_tokens: int


def _normalise(text: str) -> str:
    """Collapse whitespace and case for comparison only. Never stored."""
    return re.sub(r"[^a-z0-9 ]", "", " ".join(text.lower().split()))


def _shingles(text: str, size: int = 8) -> set[str]:
    """Overlapping word windows, used to detect near-duplicate passages.

    Exact matching is useless here: the HIGHLIGHTS copy of a sentence usually
    differs by a few words from the full-section version. Shingle overlap
    catches that; string equality does not.
    """
    words = _normalise(text).split()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def is_near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    """True when two passages say substantially the same thing."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / min(len(sa), len(sb))
    return overlap >= threshold


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _make_source(marker: int, hit: Hit, content: str) -> Source:
    return Source(
        marker=marker,
        chunk_id=hit.chunk_id,
        content=content,
        section=hit.section,
        set_id=hit.set_id,
        drug=hit.generic_name or hit.brand_name or "unknown",
        score=hit.score,
    )


def _truncate_to_budget(text: str, tokens: int) -> str:
    """Cut to fit, on a sentence boundary where possible.

    Cutting mid-sentence would hand the model half a claim — "the dose should
    be reduced in patients with" — which is worse than not sending it at all.
    """
    limit = int(tokens * CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text.strip()
    cut = text[:limit]
    boundary = cut.rfind(". ")
    return (cut[: boundary + 1] if boundary > limit // 2 else cut).strip()


def assemble(
    hits: list[Hit],
    *,
    token_budget: int | None = None,
    max_sources: int = 8,
    dedupe_threshold: float = 0.6,
) -> AssembledContext:
    """Prepare retrieved passages for generation.

    PRIORITY ORDER WHEN THE BUDGET RUNS OUT:
    Passages are taken in reranked order — best first — and the budget is spent
    from the top. So the LOWEST-scoring passages are dropped, which is the only
    defensible rule. Any other order means the model sometimes never sees the
    single best piece of evidence.
    """
    budget = token_budget or settings.context_token_budget
    sources: list[Source] = []
    kept_text: list[str] = []
    used = 0
    duplicates = 0
    dropped = 0

    for hit in hits:
        if len(sources) >= max_sources:
            dropped += 1
            continue

        if any(is_near_duplicate(hit.content, prev, dedupe_threshold) for prev in kept_text):
            duplicates += 1
            continue

        cost = estimate_tokens(hit.content) + 30  # header and marker overhead
        if used + cost > budget:
            # ALWAYS KEEP THE TOP PASSAGE, even if it alone exceeds the budget.
            # Returning zero sources would force an abstention on a question the
            # system could answer — a worse failure than a long prompt. Truncate
            # on a sentence boundary so the passage does not end mid-claim.
            if not sources:
                content = _truncate_to_budget(hit.content, budget - 30)
                sources.append(_make_source(1, hit, content))
                kept_text.append(content)
                used = estimate_tokens(content) + 30
                continue
            dropped += 1
            continue

        sources.append(_make_source(len(sources) + 1, hit, hit.content.strip()))
        kept_text.append(hit.content)
        used += cost

    return AssembledContext(
        sources=sources,
        prompt_block=render_sources(sources),
        dropped_for_budget=dropped,
        dropped_as_duplicate=duplicates,
        estimated_tokens=used,
    )


def render_sources(sources: list[Source]) -> str:
    """The block of numbered passages that goes into the prompt."""
    parts = []
    for source in sources:
        parts.append(
            f"[{source.marker}] {source.label}\n{' '.join(source.content.split())}"
        )
    return "\n\n".join(parts)