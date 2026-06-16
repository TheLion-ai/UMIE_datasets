"""Tests for the dataset audit tool (Theme P, Task 32)."""

import os
import tempfile

import cv2
import jsonlines
import numpy as np

from utils.audit_datasets import audit_datasets, write_markdown


def _write_png(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, array)


def _build_dataset(root: str) -> None:
    """Create a synthetic processed dataset with a duplicate pair, a blank and an undersized image."""
    images_dir = os.path.join(root, "00_synthetic", "CT", "Images")
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))  # non-blank pattern
    _write_png(os.path.join(images_dir, "00_0_001_1.png"), gradient)
    _write_png(os.path.join(images_dir, "00_0_001_2.png"), gradient.copy())  # exact duplicate
    _write_png(os.path.join(images_dir, "00_0_002_1.png"), gradient.T)  # distinct
    _write_png(os.path.join(images_dir, "00_0_003_1.png"), np.full((64, 64), 128, np.uint8))  # blank
    _write_png(os.path.join(images_dir, "00_0_004_1.png"), np.tile(np.arange(8, dtype=np.uint8), (8, 1)))  # tiny

    jsonl = os.path.join(root, "00_synthetic", "00_synthetic.jsonl")
    with jsonlines.open(jsonl, mode="w") as writer:
        writer.write(
            {"umie_path": "x", "dataset_name": "synthetic", "modality_name": "CT", "labels": [{"Neoplasm": 1}]}
        )
        writer.write({"umie_path": "y", "dataset_name": "synthetic", "modality_name": "CT", "labels": []})


def test_audit_reports_duplicates_blank_and_undersized():
    """The audit flags the duplicate pair, the blank image and the undersized image - without changing data."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_dataset(tmp)
        report = audit_datasets(tmp)

        findings = report["datasets"]["00_synthetic"]
        assert findings["num_images"] == 5

        # the exact-duplicate pair is clustered together
        clusters = findings["duplicates"]["clusters"]
        assert any(sum(p.endswith(("00_0_001_1.png", "00_0_001_2.png")) for p in cluster) == 2 for cluster in clusters)

        # blank and undersized images are flagged
        blank = [os.path.basename(p) for p in findings["corrupt"]["blank"]]
        undersized = [os.path.basename(p) for p in findings["corrupt"]["undersized"]]
        assert "00_0_003_1.png" in blank
        assert "00_0_004_1.png" in undersized

        # original files are untouched (audit reports, never fixes)
        assert os.path.exists(os.path.join(tmp, "00_synthetic", "CT", "Images", "00_0_003_1.png"))


def test_audit_includes_distribution_report():
    """The audit embeds the Task 22 distribution report (label/modality stats)."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_dataset(tmp)
        report = audit_datasets(tmp)
        assert "distribution" in report
        assert report["distribution"]["num_datasets"] >= 1


def test_audit_markdown_written():
    """The markdown findings report is written with a per-dataset section."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_dataset(tmp)
        report = audit_datasets(tmp)
        out = os.path.join(tmp, "audit.md")
        write_markdown(report, out)
        text = open(out).read()
        assert "## 00_synthetic" in text
        assert "Near-duplicate clusters:" in text
