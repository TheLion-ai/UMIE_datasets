"""Audit already-integrated datasets against the data-quality checklist (Theme P, Task 32).

After the convention migration (Task 47/48), this runs the ``Image data quality checks.md`` checklist
over the **processed** datasets and produces a per-dataset findings report - duplicates,
corrupt/blank/undersized images, and label/modality imbalance - **without changing any data** (it
only reports; every fix is reviewed and recorded separately). It reuses the standalone Task 22
distribution report and ctx-free re-implementations of the Task 10/11 detectors so it can run over an
output tree directly, no pipeline context required.

Usage:
    python -m utils.audit_datasets <data_dir> [--out report.md]
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import cv2
import numpy as np

from utils.distribution_report import generate_distribution_report

DEFAULT_HASH_SIZE = 8  # dHash grid side (hash is hash_size**2 bits)
DEFAULT_DUPLICATE_THRESHOLD = 5  # max Hamming distance for a near-duplicate
DEFAULT_BLANK_STD = 1.0  # pixel std below which an image is considered blank
DEFAULT_MIN_SIZE = 16  # images smaller than this (either side) are flagged as undersized


def _dhash(image: np.ndarray, hash_size: int = DEFAULT_HASH_SIZE) -> Optional[int]:
    """Return a difference-hash of ``image`` as an int, or ``None`` if it cannot be hashed."""
    if image is None:
        return None
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")


def _dataset_dirs(data_dir: str) -> list[str]:
    """Return the per-dataset output folders (``<uid>_<name>``) under ``data_dir``."""
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and "_" in name
    )


def _image_paths(dataset_dir: str) -> list[str]:
    """Return every output PNG under a dataset's ``Images`` folders."""
    return sorted(glob.glob(os.path.join(dataset_dir, "**", "Images", "*.png"), recursive=True))


def _find_duplicates(image_paths: list[str], threshold: int = DEFAULT_DUPLICATE_THRESHOLD) -> list[list[str]]:
    """Cluster near-duplicate images by perceptual-hash Hamming distance (Task 10 logic, ctx-free)."""
    hashes: list[tuple[str, int]] = []
    for path in image_paths:
        h = _dhash(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        if h is not None:
            hashes.append((path, h))

    clusters: list[list[str]] = []
    assigned: set[int] = set()
    for i, (path_i, hash_i) in enumerate(hashes):
        if i in assigned:
            continue
        group = [path_i]
        for j in range(i + 1, len(hashes)):
            if j in assigned:
                continue
            if _hamming(hash_i, hashes[j][1]) <= threshold:
                group.append(hashes[j][0])
                assigned.add(j)
        if len(group) > 1:
            assigned.add(i)
            clusters.append(group)
    return clusters


def _find_corrupt(image_paths: list[str]) -> dict[str, list[str]]:
    """Flag unreadable, blank or undersized images (Task 11 logic, ctx-free)."""
    findings: dict[str, list[str]] = {"unreadable": [], "blank": [], "undersized": []}
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            findings["unreadable"].append(path)
            continue
        if min(image.shape[:2]) < DEFAULT_MIN_SIZE:
            findings["undersized"].append(path)
        if float(np.std(image)) < DEFAULT_BLANK_STD:
            findings["blank"].append(path)
    return findings


def audit_datasets(data_dir: str) -> dict:
    """Audit every dataset under ``data_dir`` and return a per-dataset findings report (Task 32).

    Args:
        data_dir: Root directory holding the processed ``<uid>_<name>`` dataset folders.

    Returns:
        dict: ``{"datasets": {name: findings}, "distribution": <Task 22 report>}``. ``findings``
        carries the image count, duplicate clusters and corrupt-image lists; nothing is modified.
    """
    report: dict = {"datasets": {}, "distribution": generate_distribution_report(data_dir)}
    for dataset_dir in _dataset_dirs(data_dir):
        name = os.path.basename(dataset_dir)
        images = _image_paths(dataset_dir)
        duplicates = _find_duplicates(images)
        corrupt = _find_corrupt(images)
        report["datasets"][name] = {
            "num_images": len(images),
            "duplicates": {"num_clusters": len(duplicates), "clusters": duplicates},
            "corrupt": corrupt,
            "num_flagged": len(corrupt["unreadable"]) + len(corrupt["blank"]) + len(corrupt["undersized"]),
        }
    return report


def write_markdown(report: dict, path: str) -> None:
    """Write a human-readable per-dataset audit findings report (issues to file, never silent fixes)."""
    lines = ["# Dataset audit findings (Task 32)", ""]
    for name, findings in report["datasets"].items():
        dup = findings["duplicates"]["num_clusters"]
        lines.append(f"## {name}")
        lines.append(f"- Images: {findings['num_images']}")
        lines.append(f"- Near-duplicate clusters: {dup}")
        lines.append(f"- Unreadable: {len(findings['corrupt']['unreadable'])}")
        lines.append(f"- Blank: {len(findings['corrupt']['blank'])}")
        lines.append(f"- Undersized: {len(findings['corrupt']['undersized'])}")
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Audit processed UMIE datasets (Task 32).")
    parser.add_argument("data_dir", help="Root directory holding the processed datasets")
    parser.add_argument("--out", default=None, help="Optional path to write a markdown findings report")
    args = parser.parse_args()
    report = audit_datasets(args.data_dir)
    if args.out:
        write_markdown(report, args.out)
        print(f"Wrote audit report to {args.out}")
    else:
        for name, findings in report["datasets"].items():
            print(
                f"{name}: {findings['num_images']} images, "
                f"{findings['duplicates']['num_clusters']} dup clusters, {findings['num_flagged']} flagged"
            )


if __name__ == "__main__":
    main()
