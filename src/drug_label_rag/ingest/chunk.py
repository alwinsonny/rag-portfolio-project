"""Chunking.

PHASE 1 gives you `fixed` — the naive strategy, on purpose.
PHASE 3 switches to `section` and adds parent-context enrichment.

Both live here from the start so changing strategy is a settings change, not a
rewrite, and an ablation is one config flip per row of your table.

WHAT THE CORPUS ACTUALLY LOOKS LIKE (measured, not assumed):

  * SPL section text contains NO NEWLINES. Zero. Across every section, in every
    record sampled. Splitting on paragraphs — the obvious approach, and the one
    most tutorials use — finds nothing and silently degrades to blind character
    windows.

  * Modern labels carry NUMBERED SUBSECTIONS: "7.1 Agents Increasing Serum
    Potassium", "7.2 Lithium", "12.1 Mechanism of Action". Coverage is 98% in
    warnings_and_cautions and 99% in use_in_specific_populations, averaging 9.8
    and 6.6 parts respectively. Each part is semantically self-contained.

  * Legacy-format labels do NOT. The `warnings` field carries subsection
    markers in only 3% of records.

So the splitter is a cascade: subsections where they exist, sentence packing
where they do not, character windows only as a last resort. That cascade IS the
Phase 3 result — real document structure rather than an approximation of it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from drug_label_rag.ingest.parse import ParsedDocument

SECTION_TITLES = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "dosage_forms_and_strengths": "Dosage forms and strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "use_in_specific_populations": "Use in specific populations",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "general_precautions": "General precautions",
    "pregnancy": "Pregnancy",
    "pediatric_use": "Paediatric use",
    "geriatric_use": "Geriatric use",
    "nursing_mothers": "Nursing mothers",
    "drug_interactions": "Drug interactions",
    "adverse_reactions": "Adverse reactions",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of action",
    "pharmacokinetics": "Pharmacokinetics",
    "pharmacodynamics": "Pharmacodynamics",
    "clinical_pharmacology": "Clinical pharmacology",
    "clinical_studies": "Clinical studies",
    "information_for_patients": "Information for patients",
    "description": "Description",
    "how_supplied": "How supplied",
}

# "7.2 Lithium" / "12.1 Mechanism of Action" — digit.digit followed by a
# capitalised word. Zero-width lookahead keeps the marker attached to its text.
SUBSECTION_RE = re.compile(r"(?=\b\d{1,2}\.\d{1,2}\s+[A-Z])")

# Sentence boundary. The protection against splitting "12.5 mg" or "( 7.2 )"
# comes from the LOOKAHEAD, not the lookbehind: a decimal point is followed by
# a digit, never a capital letter. The lookbehind therefore admits digits too,
# which matters because SPL sentences routinely end in one — "Titrate up to
# 160 mg/25 mg." Requiring a lowercase letter there silently disabled sentence
# splitting on exactly the dosing text where it matters most.
# The period is INSIDE the lookbehind, so only the whitespace is consumed and
# each sentence keeps its terminator. Splitting on the period itself strips it,
# which makes chunks read as truncated and hurts both embeddings and citations.
SENTENCE_RE = re.compile(r"(?<=[a-z0-9\)\]\"']\.)\s+(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class Chunk:
    set_id: str
    section: str
    position: int
    content: str        # original text, shown to the user
    embedded_text: str  # what gets embedded — may carry enrichment
    content_hash: str
    brand_name: str | None
    generic_name: str | None


def _hash(set_id: str, section: str, position: int, text: str) -> str:
    """Idempotency key. Includes position so two identical passages in one
    document do not collide and silently lose one."""
    return hashlib.sha256(f"{set_id}|{section}|{position}|{text}".encode()).hexdigest()


def _enrich(doc: ParsedDocument, section: str, text: str) -> str:
    """Prepend drug name and section title before embedding.

    PHASE 3. Usually the cheapest win in the project, for about ten lines of
    code. A subsection like "7.2 Lithium Increases in serum lithium
    concentrations..." contains neither the drug's name nor the words "drug
    interaction", so a query naming both would never match it. The enriched
    version does.

    Measure it on its own row of the ablation table — it typically beats
    switching embedding models.
    """
    name = doc.generic_name or doc.brand_name or doc.substance_name or ""
    title = SECTION_TITLES.get(section, section.replace("_", " ").capitalize())
    header = f"{name} — {title}".strip(" —")
    return f"{header}\n\n{text}" if header else text


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Character windows. The Phase 1 baseline, and the last-resort fallback."""
    if len(text) <= size:
        return [text] if text.strip() else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def _pack(parts: list[str], size: int, overlap: int) -> list[str]:
    """Combine small parts up to `size`, splitting any part that exceeds it.

    Subsections range from roughly 190 to 1,400 characters. Packing the small
    ones together avoids a corpus of one-sentence chunks, which retrieve badly
    because they carry too little context to embed meaningfully.
    """
    out: list[str] = []
    current = ""
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if len(part) > size:
            if current:
                out.append(current)
                current = ""
            # Too big even alone: sentence-split, then window whatever is left.
            sentences = SENTENCE_RE.split(part)
            if len(sentences) > 1:
                out.extend(_pack(sentences, size, overlap))
            else:
                out.extend(_split_fixed(part, size, overlap))
            continue
        if len(current) + len(part) + 1 <= size:
            current = f"{current} {part}" if current else part
        else:
            out.append(current)
            current = part
    if current:
        out.append(current)
    return out


def split_section(text: str, size: int, overlap: int) -> list[str]:
    """The cascade: subsections, then sentences, then character windows.

    This function is the Phase 3 result. Its shape is dictated entirely by
    measurements of the corpus, not by what usually works elsewhere.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    parts = [p for p in SUBSECTION_RE.split(text) if p.strip()]
    if len(parts) > 1:
        return _pack(parts, size, overlap)

    sentences = SENTENCE_RE.split(text)
    if len(sentences) > 1:
        return _pack(sentences, size, overlap)

    return _split_fixed(text, size, overlap)


def chunk_document(
    doc: ParsedDocument,
    *,
    strategy: str = "fixed",
    size: int = 1000,
    overlap: int = 150,
    enrich: bool = False,
) -> list[Chunk]:
    """Turn one document into chunks.

    strategy='fixed'   PHASE 1 — concatenate every section, split on characters.
                       Deliberately discards the structure the FDA provides.
    strategy='section' PHASE 3 — one chunk per section, split by the cascade
                       above. Never splits across sections.
    """
    chunks: list[Chunk] = []

    if strategy == "fixed":
        blob = " ".join(doc.sections[name] for name in doc.sections)
        for position, piece in enumerate(_split_fixed(blob, size, overlap)):
            chunks.append(
                Chunk(
                    set_id=doc.set_id,
                    section="all",
                    position=position,
                    content=piece,
                    embedded_text=_enrich(doc, "all", piece) if enrich else piece,
                    content_hash=_hash(doc.set_id, "all", position, piece),
                    brand_name=doc.brand_name,
                    generic_name=doc.generic_name,
                )
            )
        return chunks

    if strategy != "section":
        raise ValueError(f"Unknown chunk strategy {strategy!r}")

    position = 0
    for section, text in doc.sections.items():
        for piece in split_section(text, size, overlap):
            chunks.append(
                Chunk(
                    set_id=doc.set_id,
                    section=section,
                    position=position,
                    content=piece,
                    embedded_text=_enrich(doc, section, piece) if enrich else piece,
                    content_hash=_hash(doc.set_id, section, position, piece),
                    brand_name=doc.brand_name,
                    generic_name=doc.generic_name,
                )
            )
            position += 1
    return chunks