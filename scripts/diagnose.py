"""Answer three questions the peek summary could not.

    python scripts/diagnose.py

1. WHICH SECTION FIELDS ACTUALLY EXIST? I guessed wrong once already
   (warnings_and_precautions vs warnings_and_cautions). Rather than guess again,
   count every field across the corpus and let the data say.

2. DO SECTION TEXTS CONTAIN PARAGRAPH BREAKS? The chunker's paragraph splitter
   assumes blank lines exist. If SPL text is one continuous string, that splitter
   silently falls through to blind fixed-size windows and Phase 3 gains nothing.

3. ARE SPL SUBSECTION NUMBERS USABLE AS BOUNDARIES? Labels contain markers like
   "2.1 Dosing Information", "7.2 Lithium", "12.1 Mechanism of Action". If those
   are reliable, they are a far better split point than character counts —
   real document structure rather than an approximation of it.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

RAW_DIR = Path("data/raw")

# Fields that are metadata or identifiers, not prose sections.
NON_SECTION = {
    "id", "set_id", "effective_time", "version", "openfda", "spl_id",
    "spl_set_id", "spl_product_data_elements", "package_label_principal_display_panel",
    "spl_unclassified_section", "effective_time_original",
}

# "7.2 Lithium" / "12.1 Mechanism of Action" — a digit-dot-digit followed by a
# capitalised word. The lookahead keeps the marker with the text that follows.
SUBSECTION = re.compile(r"(?=\b\d{1,2}\.\d{1,2}\s+[A-Z])")


def load(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            return json.load(fh)["results"]


def text_of(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(str(v) for v in value)
    return ""


def main() -> int:
    files = sorted(RAW_DIR.glob("*.json.zip"))
    if not files:
        raise SystemExit(f"No partitions in {RAW_DIR}.")
    records = load(files[0])
    print(f"Reading {files[0].name} — {len(records)} records\n")

    # Prescription records only: those are the corpus you will actually build.
    rx = [
        r for r in records
        if (r.get("openfda", {}).get("product_type") or [""])[0].startswith(
            "HUMAN PRESCRIPTION"
        )
    ]
    print(f"{len(rx)} prescription records\n")

    # --- 1. what fields exist -----------------------------------------
    print("=" * 70)
    print("SECTION FIELDS PRESENT (prescription records, top 30)")
    print("=" * 70)
    counts: Counter[str] = Counter()
    for record in rx:
        for key, value in record.items():
            if key in NON_SECTION or not value:
                continue
            if isinstance(value, list | str):
                counts[key] += 1
    for field, n in counts.most_common(30):
        print(f"  {n:>6} ({n / len(rx) * 100:>4.0f}%)  {field}")

    # --- 2. paragraph breaks ------------------------------------------
    print()
    print("=" * 70)
    print("TEXT STRUCTURE — is there anything to split on?")
    print("=" * 70)
    sample_fields = ["adverse_reactions", "warnings_and_cautions", "warnings",
                     "use_in_specific_populations", "dosage_and_administration"]
    for field in sample_fields:
        texts = [t for r in rx if (t := text_of(r, field))]
        if not texts:
            print(f"  {field:<32} (absent)")
            continue
        # Only count breaks INSIDE an element — the joiner adds \n\n itself.
        inner = [
            t for r in rx
            if isinstance(r.get(field), list) and r[field]
            for t in [str(r[field][0])]
        ]
        with_breaks = sum(1 for t in inner if "\n\n" in t)
        with_newline = sum(1 for t in inner if "\n" in t)
        with_subsec = sum(1 for t in inner if len(SUBSECTION.split(t)) > 1)
        avg_parts = (
            sum(len(SUBSECTION.split(t)) for t in inner) / len(inner) if inner else 0
        )
        print(f"  {field}")
        print(f"    elements sampled       {len(inner)}")
        print(f"    contain blank lines    {with_breaks:>6} ({with_breaks/max(1,len(inner))*100:>3.0f}%)")
        print(f"    contain any newline    {with_newline:>6} ({with_newline/max(1,len(inner))*100:>3.0f}%)")
        print(f"    contain '7.2 Title'    {with_subsec:>6} ({with_subsec/max(1,len(inner))*100:>3.0f}%)"
              f"   avg parts {avg_parts:.1f}")

    # --- 3. does subsection splitting produce sane pieces? -------------
    print()
    print("=" * 70)
    print("SUBSECTION SPLIT — worked example")
    print("=" * 70)
    for record in rx:
        text = text_of(record, "drug_interactions")
        parts = SUBSECTION.split(text)
        if len(parts) >= 4:
            name = (record.get("openfda", {}).get("generic_name") or ["?"])[0]
            print(f"  {name}  ->  {len(parts)} parts from {len(text)} chars\n")
            for i, part in enumerate(parts[:5]):
                head = " ".join(part.split())[:110]
                print(f"    [{i}] ({len(part):>5} chars) {head}")
            break

    # --- 4. multi-value substance names --------------------------------
    print()
    print("=" * 70)
    print("COMBINATION DRUGS — substance_name with more than one value")
    print("=" * 70)
    multi = [
        r for r in rx
        if len(r.get("openfda", {}).get("substance_name") or []) > 1
    ]
    print(f"  {len(multi)} of {len(rx)} ({len(multi)/max(1,len(rx))*100:.0f}%)")
    for record in multi[:3]:
        print(f"    {record['openfda']['substance_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())