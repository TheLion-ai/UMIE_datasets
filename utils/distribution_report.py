"""Dataset distribution report over the processed UMIE JSONL files (Task 22).

This is a standalone *utility* (not a pipeline step). Given a ``data_dir`` containing one or
more processed datasets, each laid out as ``{data_dir}/{uid}_{name}/{uid}_{name}.jsonl``, it
reads every dataset JSONL and computes:

- per-dataset and global label frequencies (labels are lists of ``{radlex_name: grade}`` dicts),
- a modality breakdown by ``modality_name``,
- per-dataset class-imbalance ratios (``max_count / min_count`` over that dataset's labels),
- a segmentation-mask size distribution (nonzero-pixel counts per mask) for masks that exist
  and are readable with OpenCV.

It writes a markdown report and a CSV next to ``output_path`` (defaulting to
``data_dir/distribution_report.md`` and ``.../distribution_report.csv``) and returns the
computed stats dict so it is directly unit-testable without asserting on files on disk.

Robustness: missing or unreadable JSONL / mask files are skipped, never crash the run, and the
function never reprocesses images beyond reading mask pixel arrays.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import Counter
from typing import Any, Optional

import cv2  # type: ignore[import-untyped]
import numpy as np


def generate_distribution_report(data_dir: str, output_path: Optional[str] = None) -> dict:
    """Compute label / modality / imbalance / mask-size distributions over a data dir.

    Args:
        data_dir (str): Directory containing ``{uid}_{name}/{uid}_{name}.jsonl`` datasets.
        output_path (Optional[str]): Base path for the markdown report; the CSV is written
            alongside with a ``.csv`` extension. Defaults to ``data_dir/distribution_report.md``.
    Returns:
        dict: The computed statistics (per-dataset + global label counts, modality breakdown,
        class-imbalance ratios, and mask-size distribution).
    """
    per_dataset: dict[str, dict[str, Any]] = {}
    global_labels: Counter = Counter()
    global_modalities: Counter = Counter()
    global_mask_sizes: list[int] = []
    total_records = 0

    for jsonl_path in _find_dataset_jsonls(data_dir):
        dataset_key = os.path.basename(os.path.dirname(jsonl_path))
        records = _read_jsonl(jsonl_path)
        total_records += len(records)

        labels = _count_labels(records)
        modalities = _count_modalities(records)
        mask_sizes = _mask_sizes(records, data_dir)

        global_labels.update(labels)
        global_modalities.update(modalities)
        global_mask_sizes.extend(mask_sizes)

        per_dataset[dataset_key] = {
            "num_records": len(records),
            "label_counts": dict(labels),
            "modality_counts": dict(modalities),
            "class_imbalance_ratio": _imbalance_ratio(labels),
            "mask_size_distribution": _summarize_sizes(mask_sizes),
        }

    stats: dict[str, Any] = {
        "data_dir": data_dir,
        "num_datasets": len(per_dataset),
        "total_records": total_records,
        "per_dataset": per_dataset,
        "global": {
            "label_counts": dict(global_labels),
            "modality_counts": dict(global_modalities),
            "class_imbalance_ratio": _imbalance_ratio(global_labels),
            "mask_size_distribution": _summarize_sizes(global_mask_sizes),
        },
    }

    md_path = output_path or os.path.join(data_dir, "distribution_report.md")
    csv_path = os.path.splitext(md_path)[0] + ".csv"
    try:
        _write_markdown(stats, md_path)
        _write_csv(stats, csv_path)
    except OSError:
        # Writing reports is best-effort; never let an unwritable path break the computation.
        pass

    return stats


def _find_dataset_jsonls(data_dir: str) -> list[str]:
    """Find every ``{uid}_{name}/{uid}_{name}.jsonl`` under ``data_dir``.

    Args:
        data_dir (str): Root data directory.
    Returns:
        list[str]: Sorted list of dataset JSONL paths whose name matches their folder.
    """
    found: list[str] = []
    for jsonl_path in sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl"))):
        folder = os.path.basename(os.path.dirname(jsonl_path))
        if os.path.basename(jsonl_path) == f"{folder}.jsonl":
            found.append(jsonl_path)
    return found


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of records, skipping unreadable files / lines.

    Args:
        path (str): Path to the JSONL file.
    Returns:
        list[dict]: Parsed records (empty when the file is missing or unreadable).
    """
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _count_labels(records: list[dict]) -> Counter:
    """Count RadLex label occurrences across records.

    ``labels`` is a list of ``{radlex_name: grade}`` dicts; each radlex name counts once per
    record it appears on.

    Args:
        records (list[dict]): Dataset records.
    Returns:
        Counter: ``{radlex_name: count}``.
    """
    counter: Counter = Counter()
    for record in records:
        for label in record.get("labels", []) or []:
            if isinstance(label, dict):
                for radlex_name in label.keys():
                    counter[str(radlex_name)] += 1
    return counter


def _count_modalities(records: list[dict]) -> Counter:
    """Count modality occurrences by ``modality_name``.

    Args:
        records (list[dict]): Dataset records.
    Returns:
        Counter: ``{modality_name: count}``.
    """
    counter: Counter = Counter()
    for record in records:
        modality_name = record.get("modality_name")
        if modality_name:
            counter[str(modality_name)] += 1
    return counter


def _imbalance_ratio(counts: Counter) -> Optional[float]:
    """Compute the class-imbalance ratio (max_count / min_count) over label counts.

    Args:
        counts (Counter): Label counts.
    Returns:
        Optional[float]: The ratio, or ``None`` when fewer than two labels are present.
    """
    values = [value for value in counts.values() if value > 0]
    if len(values) < 2:
        return None
    return round(max(values) / min(values), 4)


def _mask_sizes(records: list[dict], data_dir: str) -> list[int]:
    """Read nonzero-pixel counts for each record's mask, when the mask exists and is readable.

    Args:
        records (list[dict]): Dataset records.
        data_dir (str): Root data dir, used to resolve relative ``mask_path`` values.
    Returns:
        list[int]: Nonzero-pixel counts (one per readable mask).
    """
    sizes: list[int] = []
    for record in records:
        mask_path = record.get("mask_path")
        if not mask_path:
            continue
        resolved = mask_path if os.path.isabs(mask_path) else os.path.join(data_dir, mask_path)
        if not os.path.exists(resolved):
            continue
        try:
            mask = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
        except Exception:
            continue
        if mask is None:
            continue
        sizes.append(int(np.count_nonzero(mask)))
    return sizes


def _summarize_sizes(sizes: list[int]) -> dict[str, Any]:
    """Summarise a list of nonzero-pixel counts.

    Args:
        sizes (list[int]): Nonzero-pixel counts.
    Returns:
        dict[str, Any]: ``count`` plus ``min`` / ``max`` / ``mean`` / ``median`` (None when empty).
    """
    if not sizes:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(sizes, dtype=float)
    return {
        "count": len(sizes),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 2),
        "median": round(float(np.median(array)), 2),
    }


def _write_markdown(stats: dict, path: str) -> None:
    """Write the human-readable markdown distribution report.

    Args:
        stats (dict): The computed statistics.
        path (str): Destination markdown path.
    """
    lines: list[str] = []
    lines.append("# Dataset Distribution Report")
    lines.append("")
    lines.append(f"- Data dir: `{stats['data_dir']}`")
    lines.append(f"- Datasets: {stats['num_datasets']}")
    lines.append(f"- Total records: {stats['total_records']}")
    lines.append("")

    lines.append("## Global label frequencies")
    lines.append("")
    lines.append("| RadLex label | Count |")
    lines.append("| --- | --- |")
    for name, count in sorted(stats["global"]["label_counts"].items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {count} |")
    lines.append("")

    lines.append("## Global modality breakdown (by modality_name)")
    lines.append("")
    lines.append("| Modality | Count |")
    lines.append("| --- | --- |")
    for name, count in sorted(stats["global"]["modality_counts"].items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append(f"Global class-imbalance ratio: {stats['global']['class_imbalance_ratio']}")
    lines.append(f"Global mask-size distribution: {stats['global']['mask_size_distribution']}")
    lines.append("")

    lines.append("## Per-dataset")
    lines.append("")
    for dataset_key, info in sorted(stats["per_dataset"].items()):
        lines.append(f"### {dataset_key}")
        lines.append("")
        lines.append(f"- Records: {info['num_records']}")
        lines.append(f"- Class-imbalance ratio: {info['class_imbalance_ratio']}")
        lines.append(f"- Label counts: {info['label_counts']}")
        lines.append(f"- Modality counts: {info['modality_counts']}")
        lines.append(f"- Mask-size distribution: {info['mask_size_distribution']}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_csv(stats: dict, path: str) -> None:
    """Write a flat per-dataset, per-label CSV alongside the markdown report.

    Args:
        stats (dict): The computed statistics.
        path (str): Destination CSV path.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "dataset", "metric", "key", "value"])
        for name, count in sorted(stats["global"]["label_counts"].items()):
            writer.writerow(["global", "", "label_count", name, count])
        for name, count in sorted(stats["global"]["modality_counts"].items()):
            writer.writerow(["global", "", "modality_count", name, count])
        writer.writerow(["global", "", "class_imbalance_ratio", "", stats["global"]["class_imbalance_ratio"]])
        for dataset_key, info in sorted(stats["per_dataset"].items()):
            writer.writerow(["dataset", dataset_key, "num_records", "", info["num_records"]])
            writer.writerow(["dataset", dataset_key, "class_imbalance_ratio", "", info["class_imbalance_ratio"]])
            for name, count in sorted(info["label_counts"].items()):
                writer.writerow(["dataset", dataset_key, "label_count", name, count])
            for name, count in sorted(info["modality_counts"].items()):
                writer.writerow(["dataset", dataset_key, "modality_count", name, count])
