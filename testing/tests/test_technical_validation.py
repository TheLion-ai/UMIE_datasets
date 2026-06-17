"""Tests for the Technical Validation aggregator (Theme N, Task 44)."""

import os
import tempfile

import cv2
import jsonlines
import numpy as np

from utils.technical_validation import build_technical_validation, write_markdown


def _build_dataset(root: str) -> None:
    images_dir = os.path.join(root, "00_synthetic", "CT", "Images")
    os.makedirs(images_dir, exist_ok=True)
    gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    cv2.imwrite(os.path.join(images_dir, "00_0_001_1.png"), gradient)
    cv2.imwrite(os.path.join(images_dir, "00_0_001_2.png"), gradient.copy())  # duplicate
    cv2.imwrite(os.path.join(images_dir, "00_0_002_1.png"), np.full((64, 64), 128, np.uint8))  # blank
    with jsonlines.open(os.path.join(root, "00_synthetic", "00_synthetic.jsonl"), mode="w") as writer:
        writer.write({"umie_path": "x", "dataset_name": "synthetic", "modality_name": "CT", "labels": []})


def test_technical_validation_aggregates_totals():
    """The summary aggregates per-dataset images, duplicate clusters and flagged images."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_dataset(tmp)
        summary = build_technical_validation(tmp)
        assert summary["totals"]["datasets"] == 1
        assert summary["totals"]["images"] == 3
        assert summary["totals"]["duplicate_clusters"] >= 1
        assert summary["totals"]["flagged_images"] >= 1  # the blank image
        assert summary["per_dataset"][0]["dataset"] == "00_synthetic"


def test_technical_validation_markdown():
    """The rendered markdown has the Technical Validation heading and the per-dataset table."""
    with tempfile.TemporaryDirectory() as tmp:
        _build_dataset(tmp)
        summary = build_technical_validation(tmp)
        out = os.path.join(tmp, "tv.md")
        write_markdown(summary, out)
        text = open(out).read()
        assert "## Technical Validation" in text
        assert "| Dataset | Images |" in text
        assert "00_synthetic" in text
