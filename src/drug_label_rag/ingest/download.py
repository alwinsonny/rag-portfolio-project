"""Download the openFDA drug labelling corpus.

PHASE 1, task 1. Run once:

    python -m drug_label_rag.ingest.download            # all partitions
    python -m drug_label_rag.ingest.download --limit 2  # just the first two

WHY A SCRIPT AND NOT curl BY HAND:
The partition file list changes with every openFDA update, so hardcoded
filenames rot. This reads the current index, records the export date, and writes
a manifest — which is what makes your evaluation numbers reproducible. A number
measured against "the openFDA corpus" means nothing; a number measured against
export 2026-06-06, partitions 1-3, is a real claim.

Downloads are skipped if the file already exists with the right size, so this is
safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx

INDEX_URL = "https://api.fda.gov/download.json"
RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.json"


def fetch_index(client: httpx.Client) -> dict[str, Any]:
    """Read the current download index and pull out the drug/label section."""
    response = client.get(INDEX_URL, timeout=60.0)
    response.raise_for_status()
    body = response.json()
    try:
        return body["results"]["drug"]["label"]
    except KeyError:  # pragma: no cover - only if openFDA restructures
        print("Unexpected index shape. Top-level keys:", list(body.get("results", {})),
              file=sys.stderr)
        raise


def download_partition(client: httpx.Client, url: str, dest: Path) -> bool:
    """Stream one partition to disk. Returns True if it was actually fetched."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.name} (already present)")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, timeout=300.0, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r  fetch {dest.name}  {pct:5.1f}%", end="", flush=True)
    tmp.rename(dest)
    print(f"\r  fetch {dest.name}  done ({written / 1e6:.1f} MB)")
    return True


def verify(path: Path) -> int:
    """Open the zip and count records, so a truncated download fails loudly here."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            data = json.load(fh)
    return len(data.get("results", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the openFDA drug label corpus.")
    parser.add_argument("--limit", type=int, help="Only fetch the first N partitions")
    parser.add_argument("--out", default=str(RAW_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        index = fetch_index(client)
        export_date = index.get("export_date", "unknown")
        partitions = index.get("partitions", [])
        total_records = index.get("total_records", "unknown")

        print(f"openFDA drug/label")
        print(f"  export date   {export_date}")
        print(f"  total records {total_records}")
        print(f"  partitions    {len(partitions)}")
        print()

        selected = partitions[: args.limit] if args.limit else partitions
        fetched: list[dict[str, Any]] = []

        for part in selected:
            url = part["file"]
            dest = out_dir / Path(url).name
            download_partition(client, url, dest)
            count = verify(dest)
            fetched.append({
                "file": dest.name,
                "url": url,
                "records_in_file": count,
                "size_mb": part.get("size_mb"),
            })

    manifest = {
        "source": "openFDA drug/label",
        "export_date": export_date,
        "total_records_available": total_records,
        "partitions_available": len(partitions),
        "partitions_downloaded": len(fetched),
        "files": fetched,
        "note": (
            "Data courtesy of the U.S. Food and Drug Administration. "
            "The FDA does not endorse this tool. Public domain."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Downloaded {len(fetched)} partition(s), "
          f"{sum(f['records_in_file'] for f in fetched)} records total.")
    print(f"Manifest written to {out_dir / 'manifest.json'}")
    print()
    print("PUT THE EXPORT DATE IN YOUR README:", export_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())