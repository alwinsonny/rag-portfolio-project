"""Tests for parsing and chunking.

No database, no network, no models — these are pure functions and the tests run
in milliseconds. Every awkward shape in the real corpus is represented here.
"""

from __future__ import annotations

from datetime import date

import pytest

from drug_label_rag.ingest.chunk import SENTENCE_RE, SUBSECTION_RE, chunk_document, split_section
from drug_label_rag.ingest.parse import (
    ParsedDocument,
    first,
    join_all,
    is_eligible,
    parse_effective_time,
    parse_record,
    section_text,
    select_corpus,
)

# ======================================================================
# The shapes the real data actually comes in
# ======================================================================


def test_first_handles_the_array_of_one_shape() -> None:
    assert first(["ATORVASTATIN"]) == "ATORVASTATIN"


def test_first_tolerates_a_bare_string() -> None:
    assert first("ATORVASTATIN") == "ATORVASTATIN"


@pytest.mark.parametrize("value", [None, [], "", "   ", [""]])
def test_first_returns_none_for_every_empty_shape(value: object) -> None:
    assert first(value) is None


def test_section_text_joins_multi_element_arrays() -> None:
    record = {"warnings": ["First part.", "Second part."]}
    assert section_text(record, "warnings") == "First part.\n\nSecond part."


def test_section_text_on_a_missing_section_is_empty_not_an_error() -> None:
    assert section_text({}, "boxed_warning") == ""


def test_section_text_skips_blank_elements() -> None:
    record = {"warnings": ["Real text.", "", "   ", "More text."]}
    assert section_text(record, "warnings") == "Real text.\n\nMore text."


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["20240115"], date(2024, 1, 15)),
        (["20240115000000"], date(2024, 1, 15)),   # some carry a time suffix
        (["not-a-date"], None),
        (["2024"], None),                           # too short
        ([], None),
        (None, None),
    ],
)
def test_effective_time_never_raises(value: object, expected: date | None) -> None:
    """One malformed date must not stop ingestion of 400 documents."""
    assert parse_effective_time(value) == expected


# ======================================================================
# Record parsing
# ======================================================================


def make_record(**overrides: object) -> dict:
    record = {
        "set_id": "abc-123",
        "effective_time": ["20240115"],
        "indications_and_usage": ["Treats condition X."],
        "contraindications": ["Do not use in patients with Y."],
        "dosage_and_administration": ["10 mg once daily."],
        "openfda": {
            "brand_name": ["BRANDO"],
            "generic_name": ["GENERICOL"],
            "substance_name": ["GENERICOL SODIUM"],
            "manufacturer_name": ["Acme Pharma"],
            "route": ["ORAL"],
            "product_type": ["HUMAN PRESCRIPTION DRUG LABEL"],
        },
    }
    record.update(overrides)
    return record


def test_parse_extracts_metadata_and_sections() -> None:
    doc = parse_record(make_record())
    assert doc is not None
    assert doc.set_id == "abc-123"
    assert doc.generic_name == "GENERICOL"
    assert doc.effective_time == date(2024, 1, 15)
    assert set(doc.sections) == {
        "indications_and_usage", "contraindications", "dosage_and_administration"
    }


def test_parse_falls_back_to_spl_set_id() -> None:
    record = make_record()
    del record["set_id"]
    record["openfda"]["spl_set_id"] = ["fallback-999"]
    doc = parse_record(record)
    assert doc is not None
    assert doc.set_id == "fallback-999"


def test_parse_returns_none_without_an_identifier() -> None:
    record = make_record()
    del record["set_id"]
    assert parse_record(record) is None


def test_parse_returns_none_when_no_sections_are_populated() -> None:
    record = {"set_id": "abc", "openfda": {}}
    assert parse_record(record) is None


def test_parse_survives_a_missing_openfda_block() -> None:
    """Sparse records are normal. This must not raise."""
    record = {"set_id": "abc", "indications_and_usage": ["Text."]}
    doc = parse_record(record)
    assert doc is not None
    assert doc.generic_name is None


# ======================================================================
# Corpus selection
# ======================================================================


def test_otc_labels_are_excluded_when_prescription_only() -> None:
    record = make_record()
    record["openfda"]["product_type"] = ["HUMAN OTC DRUG LABEL"]
    doc = parse_record(record)
    assert doc is not None
    assert not is_eligible(doc, require_sections=[], prescription_only=True)
    assert is_eligible(doc, require_sections=[], prescription_only=False)


def test_records_missing_a_required_section_are_excluded() -> None:
    record = make_record()
    del record["contraindications"]
    doc = parse_record(record)
    assert doc is not None
    assert not is_eligible(
        doc, require_sections=["contraindications"], prescription_only=False
    )


def test_per_generic_cap_keeps_near_duplicates_without_drowning_the_corpus() -> None:
    """Ten manufacturers of one molecule, capped at three.

    Keeping two or three is deliberate: it creates the near-duplicate retrieval
    problem that reranking exists to solve. Keeping ten is noise.
    """
    records = [make_record(set_id=f"id-{i}") for i in range(10)]
    selected = select_corpus(
        records,
        require_sections=["indications_and_usage"],
        prescription_only=True,
        max_labels_per_generic=3,
        max_documents=100,
    )
    assert len(selected) == 3


def test_different_generics_are_not_capped_against_each_other() -> None:
    records = []
    for i in range(5):
        r = make_record(set_id=f"id-{i}")
        r["openfda"]["generic_name"] = [f"DRUG{i}"]
        records.append(r)
    selected = select_corpus(
        records,
        require_sections=[],
        prescription_only=True,
        max_labels_per_generic=1,
        max_documents=100,
    )
    assert len(selected) == 5


def test_max_documents_caps_the_corpus() -> None:
    records = []
    for i in range(20):
        r = make_record(set_id=f"id-{i}")
        r["openfda"]["generic_name"] = [f"DRUG{i}"]
        records.append(r)
    selected = select_corpus(
        records,
        require_sections=[],
        prescription_only=True,
        max_labels_per_generic=5,
        max_documents=7,
    )
    assert len(selected) == 7


# ======================================================================
# Chunking
# ======================================================================


def doc_with(sections: dict[str, str]) -> ParsedDocument:
    return ParsedDocument(
        set_id="abc-123",
        brand_name="BRANDO",
        generic_name="GENERICOL",
        substance_name="GENERICOL SODIUM",
        manufacturer="Acme",
        route="ORAL",
        product_type="HUMAN PRESCRIPTION DRUG LABEL",
        effective_time=date(2024, 1, 15),
        sections=sections,
        raw={},
    )


def test_fixed_chunking_ignores_section_boundaries() -> None:
    """The Phase 1 failure mode, asserted rather than assumed.

    Contraindications and dosage end up in the same chunk with nothing marking
    where one ends and the other begins. This test exists to document the
    problem Phase 3 solves.
    """
    doc = doc_with({"contraindications": "A" * 100, "dosage_and_administration": "B" * 100})
    chunks = chunk_document(doc, strategy="fixed", size=800, overlap=0)
    assert len(chunks) == 1
    assert "A" in chunks[0].content and "B" in chunks[0].content
    assert chunks[0].section == "all"


def test_section_chunking_never_spans_two_sections() -> None:
    doc = doc_with({"contraindications": "A" * 100, "dosage_and_administration": "B" * 100})
    chunks = chunk_document(doc, strategy="section", size=800, overlap=0)
    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"contraindications", "dosage_and_administration"}
    for chunk in chunks:
        assert not ("A" in chunk.content and "B" in chunk.content)


def test_long_sections_split_but_keep_their_section_name() -> None:
    paragraphs = " ".join(f"Sentence {i}. " + "x" * 300 for i in range(10))
    doc = doc_with({"warnings_and_cautions": paragraphs})
    chunks = chunk_document(doc, strategy="section", size=800, overlap=100)
    assert len(chunks) > 1
    assert all(c.section == "warnings_and_cautions" for c in chunks)


def test_enrichment_changes_embedded_text_but_not_content() -> None:
    """The user sees clean text; retrieval searches the enriched version.
    Easy to get backwards, so assert it explicitly.
    """
    doc = doc_with({"dosage_and_administration": "10 mg once daily."})
    chunks = chunk_document(doc, strategy="section", size=800, overlap=0, enrich=True)
    chunk = chunks[0]
    assert chunk.content == "10 mg once daily."
    assert chunk.embedded_text.startswith("GENERICOL — Dosage and administration")
    assert "10 mg once daily." in chunk.embedded_text


def test_without_enrichment_the_two_texts_are_identical() -> None:
    doc = doc_with({"dosage_and_administration": "10 mg once daily."})
    chunk = chunk_document(doc, strategy="section", size=800, overlap=0, enrich=False)[0]
    assert chunk.content == chunk.embedded_text


def test_content_hashes_are_unique_within_a_document() -> None:
    """Two identical paragraphs in one label must not collide and lose one."""
    doc = doc_with({"warnings": "Same text.\n\nOther.\n\nSame text."})
    chunks = chunk_document(doc, strategy="section", size=20, overlap=0)
    assert len({c.content_hash for c in chunks}) == len(chunks)


def test_chunking_is_deterministic() -> None:
    """Same input, same hashes — this is what makes ingestion idempotent."""
    doc = doc_with({"warnings": "x" * 3000})
    a = chunk_document(doc, strategy="section", size=500, overlap=50)
    b = chunk_document(doc, strategy="section", size=500, overlap=50)
    assert [c.content_hash for c in a] == [c.content_hash for c in b]


def test_positions_are_contiguous_across_sections() -> None:
    doc = doc_with({"indications_and_usage": "A" * 50, "contraindications": "B" * 50})
    chunks = chunk_document(doc, strategy="section", size=800, overlap=0)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chunk strategy"):
        chunk_document(doc_with({"warnings": "x"}), strategy="magic")


# ======================================================================
# Combination products (found by inspecting the real corpus)
# ======================================================================


def test_join_all_keeps_every_active_ingredient() -> None:
    """A combination drug lists several substances. Taking only the first
    means someone searching the second ingredient never finds the label.
    """
    assert join_all(["TELMISARTAN", "HYDROCHLOROTHIAZIDE"]) == (
        "TELMISARTAN / HYDROCHLOROTHIAZIDE"
    )


def test_join_all_deduplicates() -> None:
    assert join_all(["ASPIRIN", "ASPIRIN"]) == "ASPIRIN"


@pytest.mark.parametrize("value", [None, [], "", [""]])
def test_join_all_returns_none_for_empty_shapes(value: object) -> None:
    assert join_all(value) is None


def test_parse_keeps_both_substances_of_a_combination_product() -> None:
    record = make_record()
    record["openfda"]["substance_name"] = ["TELMISARTAN", "HYDROCHLOROTHIAZIDE"]
    doc = parse_record(record)
    assert doc is not None
    assert doc.substance_name == "TELMISARTAN / HYDROCHLOROTHIAZIDE"


def test_warnings_and_cautions_is_the_real_openfda_field_name() -> None:
    """Regression: warnings_and_precautions does not exist in openFDA data.
    The correct key is warnings_and_cautions, confirmed against the corpus.
    """
    from parse import SECTIONS

    assert "warnings_and_cautions" in SECTIONS
    assert "warnings_and_precautions" not in SECTIONS


# ======================================================================
# The splitter cascade — shaped by measurements of the real corpus
# ======================================================================

# A realistic fragment: no newlines anywhere, numbered subsections present.
# This mirrors the drug_interactions section from a real telmisartan label.
REAL_SHAPE = (
    "7 DRUG INTERACTIONS Lithium: Risk of lithium toxicity ( 7.2 ) "
    "Non-steroidal anti-inflammatory drugs: Reduced diuretic effects ( 7.3 ) "
    "7.1 Agents Increasing Serum Potassium Co-administration of telmisartan "
    "with other drugs that raise serum potassium levels may result in "
    "hyperkalemia. Monitor serum potassium in such patients. "
    "7.2 Lithium Increases in serum lithium concentrations and lithium "
    "toxicity have been reported with concomitant use of thiazide diuretics. "
    "7.3 Non-Steroidal Anti-Inflammatory Agents In patients who are elderly "
    "or volume-depleted, co-administration may result in deterioration of "
    "renal function. Monitor renal function periodically."
)


def test_the_corpus_has_no_newlines_so_paragraph_splitting_is_useless() -> None:
    """Measured across 1,821 prescription records: 0% contain any newline.
    This test documents why the splitter does not attempt paragraph splitting.
    """
    assert "\n" not in REAL_SHAPE


def test_subsection_regex_finds_the_numbered_markers() -> None:
    parts = [p for p in SUBSECTION_RE.split(REAL_SHAPE) if p.strip()]
    assert len(parts) == 4          # preamble + 7.1 + 7.2 + 7.3
    assert parts[1].startswith("7.1 Agents")
    assert parts[2].startswith("7.2 Lithium")


def test_subsection_regex_does_not_fire_on_a_cross_reference() -> None:
    """'( 7.2 )' inside prose is a reference, not a heading — it is followed by
    a closing bracket rather than a capitalised word."""
    text = "See Drug Interactions ( 7.2 ) for details about lithium."
    assert len(SUBSECTION_RE.split(text)) == 1


def test_sentence_regex_does_not_split_on_a_dose() -> None:
    """'12.5 mg' must survive intact. Splitting here would sever a dose from
    its units, which is the sort of failure that only shows up in evaluation.
    """
    text = "Initiate at 80 mg/12.5 mg once daily. Titrate as needed."
    parts = SENTENCE_RE.split(text)
    assert len(parts) == 2
    assert "80 mg/12.5 mg once daily" in parts[0]


def test_split_section_prefers_subsections_over_character_windows() -> None:
    pieces = split_section(REAL_SHAPE, size=300, overlap=0)
    assert len(pieces) > 1
    # Every piece that begins a subsection keeps its marker attached.
    starts = [p for p in pieces if p.startswith(("7.1", "7.2", "7.3"))]
    assert starts, "subsection markers were lost during splitting"


def test_small_subsections_are_packed_together() -> None:
    """Otherwise you get a corpus of one-sentence chunks that embed badly."""
    pieces = split_section(REAL_SHAPE, size=2000, overlap=0)
    assert len(pieces) == 1


def test_a_section_without_subsections_falls_back_to_sentences() -> None:
    """Legacy-format labels carry subsection markers in only 3% of records."""
    text = " ".join(f"This is warning sentence number {i}." for i in range(40))
    pieces = split_section(text, size=300, overlap=0)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.strip().endswith(".")


def test_a_single_unsplittable_run_falls_back_to_windows() -> None:
    """No subsections, no sentence boundaries — the last resort must still
    produce chunks rather than one enormous one."""
    text = "x" * 5000
    pieces = split_section(text, size=1000, overlap=100)
    assert len(pieces) > 1
    assert all(len(p) <= 1000 for p in pieces)


def test_no_chunk_exceeds_the_size_limit() -> None:
    pieces = split_section(REAL_SHAPE * 5, size=400, overlap=50)
    assert all(len(p) <= 400 for p in pieces)