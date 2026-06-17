"""Technical Validation suite for the Data Descriptor manuscript (Theme N, Task 44).

Aggregates the data-quality evidence that shows the corpus is useful and reliable into
publishable tables: deduplication (Task 10), corrupt/blank/undersized images (Task 11),
annotation completeness (Task 12) and cross-dataset distribution / imbalance (Task 22). It does this
by reusing the Task 32 audit (which itself reuses those detectors), so there is a single source of
truth for the numbers that appear in the *Technical Validation* section of the manuscript.

Usage:
    python -m utils.technical_validation ./data --out paper/technical_validation.md
"""

from __future__ import annotations

import argparse

from utils.audit_datasets import audit_datasets


def build_technical_validation(data_dir: str) -> dict:
    """Aggregate the audit findings into a manuscript-ready technical-validation summary (Task 44).

    Args:
        data_dir: Root directory holding the processed datasets.

    Returns:
        dict: Totals plus a per-dataset table and the embedded distribution report.
    """
    audit = audit_datasets(data_dir)
    per_dataset = []
    totals = {"datasets": 0, "images": 0, "duplicate_clusters": 0, "flagged_images": 0}
    for name, findings in audit["datasets"].items():
        dup = findings["duplicates"]["num_clusters"]
        flagged = findings["num_flagged"]
        per_dataset.append(
            {"dataset": name, "images": findings["num_images"], "duplicate_clusters": dup, "flagged_images": flagged}
        )
        totals["datasets"] += 1
        totals["images"] += findings["num_images"]
        totals["duplicate_clusters"] += dup
        totals["flagged_images"] += flagged
    return {"totals": totals, "per_dataset": per_dataset, "distribution": audit["distribution"]}


def write_markdown(summary: dict, path: str) -> None:
    """Render the technical-validation summary as a manuscript-ready markdown section."""
    totals = summary["totals"]
    lines = [
        "## Technical Validation",
        "",
        "The corpus is validated for completeness and reliability by aggregating the duplicate "
        "detection (Task 10), corrupt-image detection (Task 11), annotation-quality (Task 12) and "
        "cross-dataset distribution (Task 22) results across every integrated dataset.",
        "",
        "### Summary",
        "",
        f"- Datasets: {totals['datasets']}",
        f"- Total images: {totals['images']}",
        f"- Near-duplicate clusters: {totals['duplicate_clusters']}",
        f"- Images flagged (corrupt/blank/undersized): {totals['flagged_images']}",
        "",
        "### Per-dataset quality",
        "",
        "| Dataset | Images | Duplicate clusters | Flagged images |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["per_dataset"]:
        lines.append(f"| {row['dataset']} | {row['images']} | {row['duplicate_clusters']} | {row['flagged_images']} |")
    lines += [
        "",
        "### Baseline benchmark",
        "",
        "> _(to complete)_ At least one baseline experiment (e.g. a MICCAI-style segmentation or "
        "classification benchmark on a held-out split, using `CreateSplits`/Task 21) demonstrating "
        "the corpus's reuse value. Report the metric, split and model here.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the Technical Validation summary (Task 44).")
    parser.add_argument("data_dir", help="Root directory holding the processed datasets")
    parser.add_argument("--out", default="paper/technical_validation.md", help="Markdown output path")
    args = parser.parse_args()
    summary = build_technical_validation(args.data_dir)
    write_markdown(summary, args.out)
    print(f"Wrote technical validation summary to {args.out}")


if __name__ == "__main__":
    main()
