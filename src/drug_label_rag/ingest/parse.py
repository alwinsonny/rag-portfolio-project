"""Parse openFDA SPL records into Documents.

PHASE 1. Pure functions, no database, no network — so this is fully unit
testable and you can iterate on the filter without touching Postgres.

THREE PROPERTIES OF THIS DATA THAT SHAPE EVERY FUNCTION HERE:

1. Sections are arrays, usually of one long string. Never assume one element.
2. Not every label has every section. Absence is normal and must not crash —
   but absence is also information: a question about a section a label lacks is
   a legitimate unanswerable case for your Phase 2 set.
3. The same molecule appears many times with different text. This is the most
   useful property of the corpus, because it makes "which label did this passage
   come from" a real question — and it is why citations must resolve to a
   set_id, not a drug name.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Any

# The sections worth ingesting, in the order they should appear in a document.
# The sections worth ingesting, in the order they should appear.
#
# THE CORPUS CONTAINS TWO LABEL FORMATS, confirmed by counting fields across
# 1,821 prescription records:
#
#   Modern (Physician Labeling Rule): warnings_and_cautions (56%),
#     use_in_specific_populations (55%), dosage_forms_and_strengths (55%).
#     Carries numbered subsections — "7.2 Lithium" — at 98-99% coverage.
#
#   Legacy: warnings (39%), precautions (41%), pregnancy (77%),
#     pediatric_use (76%), geriatric_use (66%), nursing_mothers (42%).
#     Numbered subsections appear in only 3% of these.
#
# Both families are included. A label will populate one set or the other, and
# the chunker handles the structural difference rather than the parser.
#
# The *_table fields (adverse_reactions_table, clinical_pharmacology_table, and
# others, ~29-50% coverage) are deliberately EXCLUDED: they carry HTML markup
# that would pollute both embeddings and displayed citations. Extracting them
# properly is a Phase 5 stretch goal.
SECTIONS: tuple[str, ...] = (
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    # modern format
    "warnings_and_cautions",
    "use_in_specific_populations",
    # legacy format
    "warnings",
    "precautions",
    "general_precautions",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "nursing_mothers",
    # both
    "drug_interactions",
    "adverse_reactions",
    "overdosage",
    "mechanism_of_action",
    "pharmacokinetics",
    "pharmacodynamics",
    "clinical_pharmacology",
    "clinical_studies",
    "information_for_patients",
    "description",
    "how_supplied",
)


class ParsedDocument:
    """A cleaned label, ready to chunk. Deliberately not a Pydantic model —
    this is an internal value object and validation would just cost time."""

    __slots__ = (
        "set_id", "brand_name", "generic_name", "substance_name",
        "manufacturer", "route", "product_type", "effective_time",
        "sections", "raw",
    )

    def __init__(self, **kw: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kw.get(key))

    def __repr__(self) -> str:
        return (
            f"<ParsedDocument {self.set_id} {self.generic_name!r} "
            f"sections={list(self.sections)}>"
        )


def first(value: Any) -> str | None:
    """openfda metadata fields are arrays. Take the first, tolerate anything."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list) and value:
        return str(value[0]).strip() or None
    return None


def join_all(value: Any, sep: str = " / ") -> str | None:
    """Join every value, for genuinely multi-valued fields.

    substance_name on a combination product is ['TELMISARTAN',
    'HYDROCHLOROTHIAZIDE'] — taking only the first silently drops half the drug,
    and someone searching the second ingredient would never find the label.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list) and value:
        parts = [str(v).strip() for v in value if str(v).strip()]
        return sep.join(dict.fromkeys(parts)) or None
    return None


def section_text(record: dict[str, Any], name: str) -> str:
    """Join a section into one string. Handles the array-of-one shape, the
    array-of-many shape, and a bare string, without assuming which."""
    value = record.get(name)
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n\n".join(str(v).strip() for v in value if str(v).strip())
    return ""


def parse_effective_time(value: Any) -> date | None:
    """effective_time is YYYYMMDD as a string. Some are malformed; return None
    rather than raising, because one bad date must not stop ingestion."""
    raw = first(value)
    if not raw or len(raw) < 8:
        return None
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


def parse_record(record: dict[str, Any]) -> ParsedDocument | None:
    """Turn one raw SPL record into a ParsedDocument, or None if unusable."""
    set_id = record.get("set_id") or first(record.get("openfda", {}).get("spl_set_id"))
    if not set_id:
        return None

    meta = record.get("openfda", {})
    sections = {
        name: text for name in SECTIONS if (text := section_text(record, name))
    }
    if not sections:
        return None

    return ParsedDocument(
        set_id=set_id,
        brand_name=first(meta.get("brand_name")),
        generic_name=first(meta.get("generic_name")),
        substance_name=join_all(meta.get("substance_name")),
        manufacturer=first(meta.get("manufacturer_name")),
        route=first(meta.get("route")),
        product_type=first(meta.get("product_type")),
        effective_time=parse_effective_time(record.get("effective_time")),
        sections=sections,
        raw=record,
    )


def is_eligible(
    doc: ParsedDocument,
    *,
    require_sections: list[str],
    prescription_only: bool,
) -> bool:
    """Apply the corpus filter. Recorded in settings so it is reproducible."""
    if prescription_only:
        kind = (doc.product_type or "").upper()
        if not kind.startswith("HUMAN PRESCRIPTION"):
            return False
    return all(doc.sections.get(name) for name in require_sections)


def select_corpus(
    records: list[dict[str, Any]],
    *,
    require_sections: list[str],
    prescription_only: bool,
    max_labels_per_generic: int,
    max_documents: int,
) -> list[ParsedDocument]:
    """Filter and cap the corpus.

    The per-generic cap is the interesting parameter. Dozens of manufacturers
    publish labels for the same molecule. Keeping two or three is DELIBERATE —
    it creates the near-duplicate retrieval problem that reranking exists to
    solve, and it makes your citations meaningful. Keeping all forty is noise
    that will drown every other signal in your evaluation.
    """
    eligible: list[ParsedDocument] = []
    for record in records:
        doc = parse_record(record)
        if doc is None:
            continue
        if is_eligible(
            doc, require_sections=require_sections, prescription_only=prescription_only
        ):
            eligible.append(doc)

    # Longest labels first: they have the richest sections and make the harder,
    # more interesting corpus.
    eligible.sort(key=lambda d: sum(len(t) for t in d.sections.values()), reverse=True)

    per_generic: dict[str, int] = defaultdict(int)
    selected: list[ParsedDocument] = []
    for doc in eligible:
        key = (doc.generic_name or doc.substance_name or doc.set_id).lower()
        if per_generic[key] >= max_labels_per_generic:
            continue
        per_generic[key] += 1
        selected.append(doc)
        if len(selected) >= max_documents:
            break
    return selected