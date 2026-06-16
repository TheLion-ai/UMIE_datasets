"""Per-dataset 'datasheet for datasets' generator (Theme N, Task 43).

Turns the Obsidian ``Dataset template.md`` into a machine-fillable markdown datasheet generated from
a single source of truth: ``config/dataset_config.py`` (uid, modalities, label/mask tables),
``config/provenance.py`` (licence + source attribution, Task 23) and the processed JSONL stats
(record counts, per-modality and per-label breakdowns). Auto-filled fields come from config/JSONL;
manual narrative sections (demographics, known uses, quality/limitations) are stubbed for an author
to complete. The same machine-readable facts feed the HuggingFace card (Task 30).

Usage:
    python -m utils.datasheet kits23 --jsonl data/00_kits23/00_kits23.jsonl --out docs/datasheets
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Optional

import jsonlines

from config import dataset_config
from config import labels as labels_config
from config.dataset_config import DatasetArgs
from config.provenance import get_provenance


def all_dataset_args() -> dict[str, DatasetArgs]:
    """Return every configured dataset keyed by ``dataset_name`` (single source of truth)."""
    return {obj.dataset_name: obj for obj in vars(dataset_config).values() if isinstance(obj, DatasetArgs)}


def _jsonl_stats(jsonl_path: Optional[str]) -> dict:
    """Aggregate record counts, modality and label breakdowns from a processed JSONL (best-effort)."""
    if not jsonl_path or not os.path.exists(jsonl_path):
        return {"num_images": None, "num_annotated": None, "modalities": Counter(), "labels": Counter()}
    modalities: Counter = Counter()
    labels: Counter = Counter()
    num_images = 0
    num_annotated = 0
    with jsonlines.open(jsonl_path) as reader:
        for record in reader:
            num_images += 1
            if record.get("modality_name"):
                modalities[str(record["modality_name"])] += 1
            record_labels = record.get("labels") or []
            if record_labels or record.get("mask_path"):
                num_annotated += 1
            for label_dict in record_labels:
                for term in label_dict:
                    labels[term] += 1
    return {"num_images": num_images, "num_annotated": num_annotated, "modalities": modalities, "labels": labels}


def _label_rows(dataset: DatasetArgs) -> list[str]:
    """Build the label-description table rows from the dataset's RadLex labels."""
    used: list[str] = []
    for translations in dataset.labels.values():
        for label_dict in translations:
            used.extend(label_dict.keys())
    rows = []
    for radlex_name in sorted(set(used)):
        label = labels_config.label_by_name(radlex_name)
        radlex_id = label.radlex_id if label else ""
        rows.append(f"| {radlex_name} | {radlex_id} |")
    return rows


def generate_datasheet(dataset_name: str, jsonl_path: Optional[str] = None) -> str:
    """Generate a markdown datasheet for ``dataset_name`` (Task 43).

    Args:
        dataset_name: A configured dataset name (key of :func:`all_dataset_args`).
        jsonl_path: Optional path to the processed JSONL, for record/label counts.

    Returns:
        str: The datasheet markdown.

    Raises:
        KeyError: If ``dataset_name`` is not a configured dataset.
    """
    datasets = all_dataset_args()
    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset {dataset_name!r}; known: {sorted(datasets)}")
    dataset = datasets[dataset_name]
    provenance = get_provenance(dataset_name)
    stats = _jsonl_stats(jsonl_path)
    modalities = ", ".join(sorted(set(dataset.modalities.values())))

    lines = [
        f"# {dataset.dataset_uid}_{dataset_name}",
        "",
        "> Auto-generated datasheet (Task 43). Auto-filled fields come from `config/dataset_config.py`,",
        "> `config/provenance.py` and the processed JSONL; narrative sections marked _(to complete)_ are manual.",
        "",
        "## Overview",
        "",
        "| Property | Details |",
        "| --- | --- |",
        f"| **Dataset Name** | {dataset_name} |",
        f"| **UMIE ID** | {dataset.dataset_uid} |",
        f"| **Modality** | {modalities} |",
        f"| **Licence** | {provenance.license} |",
        f"| **Source Dataset** | {provenance.source_dataset} |",
        f"| **Source URL** | {provenance.source_url or '_(unknown)_'} |",
        f"| **Redistributable** | {provenance.redistributable} |",
        "",
        "## Scale",
        "",
        "| Property | Details |",
        "| --- | --- |",
        f"| **No. of Images** | {stats['num_images'] if stats['num_images'] is not None else '_(run on JSONL)_'} |",
        f"| **No. of Annotated Images** | {stats['num_annotated'] if stats['num_annotated'] is not None else '_(run on JSONL)_'} |",
        "",
        "### Modality breakdown",
        "",
    ]
    if stats["modalities"]:
        lines += ["| Modality | Images |", "| --- | --- |"]
        lines += [f"| {name} | {count} |" for name, count in sorted(stats["modalities"].items())]
    else:
        lines.append("_(run on a processed JSONL to populate)_")

    lines += [
        "",
        "## Labels & Masks",
        "",
        f"- **Has Labels:** {bool(dataset.labels)}",
        f"- **Has Masks:** {bool(dataset.masks)}",
        "",
    ]
    label_rows = _label_rows(dataset)
    if label_rows:
        lines += ["### RadLex labels used", "", "| Label (RadLex name) | RadLex id |", "| --- | --- |", *label_rows, ""]
    if stats["labels"]:
        lines += ["### Label frequencies (from JSONL)", "", "| Label | Count |", "| --- | --- |"]
        lines += [f"| {name} | {count} |" for name, count in stats["labels"].most_common()]
        lines.append("")

    lines += [
        "## Demographics & Metadata",
        "",
        "| Property | Details |",
        "| --- | --- |",
        "| **Age / Sex distribution** | _(to complete)_ |",
        "| **Scanner manufacturer(s)** | _(to complete)_ |",
        "| **De-identification** | _(to complete)_ |",
        "",
        "## Known Uses",
        "",
        "_(to complete: notable papers / benchmarks using this dataset)_",
        "",
        "## Quality & Limitations",
        "",
        "_(to complete: known issues, biases, class imbalance, artifacts — see `utils/audit_datasets.py`)_",
        "",
        "## References",
        "",
        f"- **Source citation:** {provenance.source_citation or '_(to complete)_'}",
        "",
    ]
    return "\n".join(lines)


def write_datasheet(dataset_name: str, out_dir: str, jsonl_path: Optional[str] = None) -> str:
    """Generate and write the datasheet to ``out_dir/<uid>_<name>.md``; return the path."""
    datasets = all_dataset_args()
    dataset = datasets[dataset_name]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset.dataset_uid}_{dataset_name}.md")
    with open(path, "w") as handle:
        handle.write(generate_datasheet(dataset_name, jsonl_path))
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate a per-dataset datasheet (Task 43).")
    parser.add_argument("dataset", nargs="?", help="Dataset name (omit to generate all)")
    parser.add_argument("--jsonl", default=None, help="Path to the processed JSONL for counts")
    parser.add_argument("--out", default="docs/datasheets", help="Output directory")
    args = parser.parse_args()
    names = [args.dataset] if args.dataset else sorted(all_dataset_args())
    for name in names:
        path = write_datasheet(name, args.out, args.jsonl if args.dataset else None)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
