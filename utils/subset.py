"""Query the corpus and generate downloadable subset manifests (Theme N, Task 45).

This is the backend for the subset/query web interface: it reads the existing JSONL metadata only
(no images are copied) and produces a **manifest** - a file list plus the matching JSONL slice - for a
filter over modality, body part, RadLex label (hierarchical) and licence. Hierarchical label queries
reuse the Theme L ontology (Task 36), so filtering by ``Neoplasm`` also matches every tumour subtype
below it.

The optional Flask wrapper in ``utils/subset_web.py`` exposes the same functions over HTTP.

Usage:
    python -m utils.subset data/00_kits23/00_kits23.jsonl --label Neoplasm --out subset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import jsonlines

from config import labels as labels_config
from config.provenance import get_provenance


def load_records(jsonl_paths: list[str]) -> list[dict]:
    """Load and concatenate records from one or more JSONL files."""
    records: list[dict] = []
    for path in jsonl_paths:
        with jsonlines.open(path) as reader:
            records.extend(reader)
    return records


def _record_labels(record: dict) -> set[str]:
    """Return the RadLex label terms on a record (handles both v1 flat and v2 hierarchical records)."""
    terms: set[str] = set()
    for label_dict in record.get("labels", []) or []:
        terms.update(label_dict.keys())
    annotations = record.get("annotations", {})
    for entry in annotations.get("image_level_classification", []):
        term = entry.get("radlex_label", {}).get("term")
        if term:
            terms.add(term)
    return terms


def _record_modality(record: dict) -> Optional[str]:
    """Return a record's modality (v1 ``modality_name`` or v2 ``series_level_info.modality``)."""
    return record.get("modality_name") or record.get("series_level_info", {}).get("modality")


def _record_license(record: dict) -> str:
    """Return a record's licence (explicit ``license`` field, else looked up from provenance)."""
    if record.get("license"):
        return str(record["license"])
    name = record.get("dataset_name") or record.get("dataset_source_details", {}).get("dataset_name", "")
    return get_provenance(name).license


def _label_query_terms(label: str, include_descendants: bool) -> set[str]:
    """Expand a label to itself plus, optionally, its hierarchy descendants (Task 36)."""
    terms = {label}
    if include_descendants:
        terms.update(d.radlex_name for d in labels_config.label_descendants_of(label))
    return terms


def query_subset(
    records: list[dict],
    *,
    modality: Optional[str] = None,
    label: Optional[str] = None,
    license: Optional[str] = None,
    include_label_descendants: bool = True,
) -> list[dict]:
    """Filter records by modality / RadLex label (hierarchical) / licence (Task 45).

    Args:
        records: The corpus records (v1 or v2).
        modality: Keep only this modality (e.g. ``CT``).
        label: Keep records carrying this RadLex label or, when ``include_label_descendants``, any
            of its hierarchy descendants.
        license: Keep only records under this licence.
        include_label_descendants: Expand ``label`` down the ontology hierarchy.

    Returns:
        list[dict]: The matching records (a slice of the input, not copies of any images).
    """
    label_terms = _label_query_terms(label, include_label_descendants) if label else None
    out = []
    for record in records:
        if modality and _record_modality(record) != modality:
            continue
        if label_terms is not None and not (_record_labels(record) & label_terms):
            continue
        if license and _record_license(record) != license:
            continue
        out.append(record)
    return out


def build_manifest(subset: list[dict]) -> dict:
    """Build a downloadable manifest (file list + JSONL slice) from a queried subset."""
    files = [r.get("umie_path") for r in subset if r.get("umie_path")]
    masks = [r.get("mask_path") for r in subset if r.get("mask_path")]
    return {"num_records": len(subset), "files": files, "masks": masks, "records": subset}


def write_manifest(subset: list[dict], path: str) -> str:
    """Write the manifest JSON to ``path`` and return it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(build_manifest(subset), handle, indent=2)
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Query the corpus and emit a subset manifest (Task 45).")
    parser.add_argument("jsonl", nargs="+", help="One or more dataset JSONL files")
    parser.add_argument("--modality", default=None)
    parser.add_argument("--label", default=None, help="RadLex label (matches descendants too)")
    parser.add_argument("--license", default=None)
    parser.add_argument("--no-descendants", action="store_true", help="Do not expand the label hierarchy")
    parser.add_argument("--out", default=None, help="Write the manifest JSON here")
    args = parser.parse_args()
    records = load_records(args.jsonl)
    subset = query_subset(
        records,
        modality=args.modality,
        label=args.label,
        license=args.license,
        include_label_descendants=not args.no_descendants,
    )
    if args.out:
        write_manifest(subset, args.out)
        print(f"Wrote manifest with {len(subset)} records to {args.out}")
    else:
        print(f"{len(subset)} records match")


if __name__ == "__main__":
    main()
