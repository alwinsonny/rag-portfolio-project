"""The citation verifier.

PHASE 5. The most important module in the project.

THE PRINCIPLE:
The prompt ASKS the model to cite its sources. This CHECKS that it did. Those
are completely different strengths of guarantee, and only the second one is
worth anything.

A model will happily write "[7]" when you supplied six passages. It is not lying
— it is producing text that looks like cited text, which is exactly what it was
trained to do. Asking it not to is a request. Rejecting the answer is a control.

WHAT IS CHECKED:

  1. RESOLVABLE      every [n] points at a passage that was actually supplied
  2. GROUNDED        every factual sentence carries at least one marker
  3. USED            at least one supplied passage is actually cited

Failure on any of these rejects the whole answer. The caller regenerates once,
then abstains — the same one-repair discipline as Project 01's dispatch layer.
A second failure means something systematic, not a fluke.

ON DECIDING WHAT IS "FACTUAL":
There is no perfect rule, and pretending otherwise would be worse than stating
the heuristic. A sentence is treated as factual when it makes a claim about the
drug: it contains a number, a unit, a clinical term, or a declarative verb.
Hedges, transitions and meta-sentences ("I could not find...") are exempt.

This is imperfect and you should say so in your write-up. An imperfect
deterministic check that rejects real violations beats a perfect-sounding
promise that checks nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from drug_label_rag.answer.assemble import Source

MARKER = re.compile(r"\[(\d{1,2})\]")

# Sentence boundary that survives clinical prose: "12.5 mg" and "( 7.2 )" must
# not split. Same reasoning as the chunker — the lookahead requires a capital,
# and a decimal point is always followed by a digit.
SENTENCE = re.compile(r"(?<=[a-z0-9\)\]\"']\.)\s+(?=[A-Z])")

# A sentence claiming something about the drug, rather than framing the answer.
FACTUAL_SIGNALS = re.compile(
    r"\b\d|"                                    # any number: dose, percentage, age
    r"\b(mg|mcg|ml|kg|hours?|days?|weeks?|months?|years?)\b|"
    r"\b(is|are|was|were|has|have|should|must|may|can|causes?|increases?|"
    r"decreases?|reduces?|indicated|contraindicated|recommended|reported|"
    r"observed|associated|required)\b",
    re.IGNORECASE,
)

# Framing that carries no claim and therefore needs no citation.
EXEMPT = re.compile(
    r"^\s*(i (could not|cannot|was unable)|the (labelling|label|documents?) "
    r"(does not|do not)|based on|according to|in summary|here is|note that)\b",
    re.IGNORECASE,
)


class Violation(StrEnum):
    UNRESOLVABLE_MARKER = "unresolvable_marker"
    UNCITED_CLAIM = "uncited_claim"
    NO_SOURCES_USED = "no_sources_used"
    EMPTY_ANSWER = "empty_answer"


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    violations: list[tuple[Violation, str]] = field(default_factory=list)
    cited_markers: set[int] = field(default_factory=set)
    factual_sentences: int = 0
    cited_sentences: int = 0

    @property
    def summary(self) -> str:
        if self.ok:
            return (
                f"verified: {self.cited_sentences}/{self.factual_sentences} factual "
                f"sentences cited, markers {sorted(self.cited_markers)}"
            )
        kinds = ", ".join(sorted({v.value for v, _ in self.violations}))
        return f"REJECTED ({len(self.violations)} violations: {kinds})"


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip list bullets so a bulleted answer is checked like prose.
        line = re.sub(r"^[-*\u2022]\s*", "", line)
        parts.extend(s.strip() for s in SENTENCE.split(line) if s.strip())
    return parts


def is_factual(sentence: str) -> bool:
    """Does this sentence make a claim that needs a source?"""
    stripped = MARKER.sub("", sentence).strip()
    if len(stripped.split()) < 4:
        return False
    if EXEMPT.match(stripped):
        return False
    return bool(FACTUAL_SIGNALS.search(stripped))


def verify(answer: str, sources: list[Source]) -> VerificationResult:
    """Check an answer against the passages that were supplied to produce it."""
    result = VerificationResult(ok=True)

    if not answer or not answer.strip():
        result.ok = False
        result.violations.append((Violation.EMPTY_ANSWER, ""))
        return result

    valid = {s.marker for s in sources}

    for sentence in split_sentences(answer):
        markers = {int(m) for m in MARKER.findall(sentence)}

        # 1. Every marker must resolve to a passage that was actually supplied.
        for marker in markers - valid:
            result.ok = False
            result.violations.append(
                (
                    Violation.UNRESOLVABLE_MARKER,
                    f"[{marker}] does not exist — {len(valid)} passages were supplied: "
                    f"{sentence[:90]}",
                )
            )
        result.cited_markers |= markers & valid

        # 2. Every factual sentence must carry at least one valid marker.
        if is_factual(sentence):
            result.factual_sentences += 1
            if markers & valid:
                result.cited_sentences += 1
            else:
                result.ok = False
                result.violations.append(
                    (Violation.UNCITED_CLAIM, sentence[:120])
                )

    # 3. An answer that cites nothing at all is ungrounded by definition.
    if result.factual_sentences and not result.cited_markers:
        result.ok = False
        result.violations.append((Violation.NO_SOURCES_USED, ""))

    return result


def repair_instruction(result: VerificationResult) -> str:
    """Feedback for the single regeneration attempt.

    Specific beats general: telling the model exactly which sentence was
    uncited works, telling it to "cite better" does not.
    """
    lines = ["Your previous answer was rejected. Fix these problems:"]
    for violation, detail in result.violations[:6]:
        match violation:
            case Violation.UNRESOLVABLE_MARKER:
                lines.append(f"- You cited a passage that was not provided: {detail}")
            case Violation.UNCITED_CLAIM:
                lines.append(f"- This claim has no source marker: \"{detail}\"")
            case Violation.NO_SOURCES_USED:
                lines.append("- You cited nothing. Every factual claim needs a marker.")
            case Violation.EMPTY_ANSWER:
                lines.append("- You produced no answer.")
    lines.append(
        "Use only the numbered passages provided. If they do not answer the "
        "question, say so plainly instead of guessing."
    )
    return "\n".join(lines)