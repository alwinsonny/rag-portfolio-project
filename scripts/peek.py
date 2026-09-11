"""Look at the corpus before you write the parser.

PHASE 1, task 2. This is the hour that saves an evening.

    python scripts/peek.py                    # summary across a partition
    python scripts/peek.py --record 0         # one full record
    python scripts/peek.py --section warnings_and_precautions --record 3

You are looking for four things:
  1. Which sections exist, and how often. (Many labels are sparse.)
  2. How long sections actually are. (This drives your chunk size.)
  3. What the openfda metadata block contains, and what is missing from it.
  4. How the same molecule appears across manufacturers.

Do not skip this. Every chunking decision in Phase 3 is a decision about text
you should have read first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

RAW_DIR = Path("data/raw")

# The sections the manual says matter. Anything outside this list is ballast.
KEY_SECTIONS = [
    "indications_and_usage",
    "contraindications",
    "dosage_and_administration",
    "warnings_and_cautions",
    "boxed_warning",
    "drug_interactions",
    "adverse_reactions",
    "use_in_specific_populations",
    "description",
    "clinical_pharmacology",
    "warnings"
]


def load(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            return json.load(fh)["results"]


def first_partition() -> Path:
    files = sorted(RAW_DIR.glob("*.json.zip"))
    if not files:
        raise SystemExit(f"No partitions in {RAW_DIR}. Run the downloader first.")
    return files[0]


def text_of(record: dict[str, Any], section: str) -> str:
    """Sections are arrays, usually of one long string. Never assume one element."""
    value = record.get(section)
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return "\n\n".join(str(v) for v in value)


def summarise(records: list[dict[str, Any]]) -> None:
    print(f"{len(records)} records in this partition\n")

    print("SECTION COVERAGE AND LENGTH")
    print(f"{'section':<32} {'present':>8} {'%':>6} {'median':>8} {'p90':>8} {'max':>8}")
    for section in KEY_SECTIONS:
        lengths = [len(text_of(r, section)) for r in records]
        present = [n for n in lengths if n > 0]
        if not present:
            print(f"{section:<32} {0:>8} {0:>5.0f}% {'-':>8} {'-':>8} {'-':>8}")
            continue
        present.sort()
        p90 = present[int(len(present) * 0.9) - 1]
        print(f"{section:<32} {len(present):>8} {len(present)/len(records)*100:>5.0f}% "
              f"{statistics.median(present):>8.0f} {p90:>8} {max(present):>8}")

    print("\nPRODUCT TYPE")
    for kind, n in Counter(
        (r.get("openfda", {}).get("product_type") or ["(missing)"])[0] for r in records
    ).most_common(5):
        print(f"  {n:>6}  {kind}")

    print("\nROUTE (top 8)")
    for route, n in Counter(
        (r.get("openfda", {}).get("route") or ["(missing)"])[0] for r in records
    ).most_common(8):
        print(f"  {n:>6}  {route}")

    print("\nMETADATA COMPLETENESS (openfda block)")
    for field in ["brand_name", "generic_name", "substance_name",
                  "manufacturer_name", "route", "product_type", "spl_set_id"]:
        have = sum(1 for r in records if r.get("openfda", {}).get(field))
        print(f"  {field:<20} {have:>6} / {len(records)}  ({have/len(records)*100:.0f}%)")

    print("\nSAME MOLECULE, MANY LABELS (top 10 generic names)")
    generics = Counter(
        (r.get("openfda", {}).get("generic_name") or ["(missing)"])[0].lower()
        for r in records
    )
    for name, n in generics.most_common(10):
        print(f"  {n:>4}  {name[:60]}")

    rich = [
        r for r in records
        if all(text_of(r, s) for s in
               ["indications_and_usage", "contraindications", "dosage_and_administration"])
        and (r.get("openfda", {}).get("product_type") or [""])[0].startswith("HUMAN PRESCRIPTION")
    ]
    boxed = [r for r in rich if text_of(r, "boxed_warning")]
    print(f"\nCANDIDATES FOR YOUR SUBSET")
    print(f"  prescription + 3 key sections populated : {len(rich)}")
    print(f"  ...of which have a boxed warning         : {len(boxed)}")


def show_record(record: dict[str, Any], section: str | None, chars: int) -> None:
    meta = record.get("openfda", {})
    print("METADATA")
    for field in ["brand_name", "generic_name", "substance_name", "manufacturer_name",
                  "route", "product_type", "spl_set_id"]:
        print(f"  {field:<20} {meta.get(field)}")
    print(f"  {'effective_time':<20} {record.get('effective_time')}")
    print(f"  {'set_id':<20} {record.get('set_id')}")

    sections = [section] if section else KEY_SECTIONS
    for name in sections:
        body = text_of(record, name)
        if not body:
            print(f"\n--- {name}  (ABSENT) ---")
            continue
        print(f"\n--- {name}  ({len(body)} chars) ---")
        print(body[:chars] + ("…" if len(body) > chars else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore the openFDA corpus.")
    parser.add_argument("--file", help="Partition zip to read (default: first found)")
    parser.add_argument("--record", type=int, help="Show one record in full")
    parser.add_argument("--section", help="Only show this section")
    parser.add_argument("--chars", type=int, default=800, help="Characters per section")
    args = parser.parse_args(argv)

    path = Path(args.file) if args.file else first_partition()
    print(f"Reading {path.name}\n")
    records = load(path)

    if args.record is not None:
        show_record(records[args.record], args.section, args.chars)
    else:
        summarise(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())