"""Unit tests for CheckMaskQuality using small synthetic PNGs (no external data)."""

import json
import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs, MaskColor
from src.steps.check_mask_quality import CheckMaskQuality


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a PipelineContext whose masks allow target colours 1 and 2."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(
            dataset_uid="99",
            dataset_name="synthetic",
            modalities={"0": "CT"},
            masks={
                "Kidney": MaskColor(source_color=1, target_color=1),
                "Neoplasm": MaskColor(source_color=2, target_color=2),
            },
        ),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _dirs(tmp: str) -> tuple:
    """Create and return the synthetic Images and Masks folders."""
    images_dir = os.path.join(tmp, "99_synthetic", "CT", "Images")
    masks_dir = os.path.join(tmp, "99_synthetic", "CT", "Masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    return images_dir, masks_dir


def _rel(path: str, tmp: str) -> str:
    """Return the umie_path (relative to the target/tmp dir) of an output file."""
    return os.path.relpath(path, tmp).replace(os.sep, "/")


def _failed_masks(report: dict, category: str) -> set:
    """Return the set of mask paths recorded under a report category."""
    members = set()
    for entry in report[category]:
        members.add(entry["mask"] if isinstance(entry, dict) else entry)
    return members


def test_classifies_dim_mismatch_out_of_vocab_and_valid():
    """A dim-mismatch mask, an out-of-vocab mask and a valid mask are each classified."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir, masks_dir = _dirs(tmp)

        # paired images, all 32x32
        for name in ("valid", "vocab", "dim"):
            cv2.imwrite(os.path.join(images_dir, f"99_0_001_{name}.png"), np.zeros((32, 32), dtype=np.uint8))

        # valid mask: allowed colours only (0 and configured target colour 1)
        valid_mask = np.zeros((32, 32), dtype=np.uint8)
        valid_mask[5:10, 5:10] = 1
        valid_path = os.path.join(masks_dir, "99_0_001_valid.png")
        cv2.imwrite(valid_path, valid_mask)

        # out-of-vocab mask: contains colour 200 which is not an allowed target
        vocab_mask = np.zeros((32, 32), dtype=np.uint8)
        vocab_mask[0, 0] = 200
        vocab_path = os.path.join(masks_dir, "99_0_001_vocab.png")
        cv2.imwrite(vocab_path, vocab_mask)

        # dim-mismatch mask: 16x16 against a 32x32 image, otherwise valid colours
        dim_mask = np.zeros((16, 16), dtype=np.uint8)
        dim_mask[0, 0] = 1
        dim_path = os.path.join(masks_dir, "99_0_001_dim.png")
        cv2.imwrite(dim_path, dim_mask)

        returned = CheckMaskQuality(_make_ctx(tmp)).transform(["unused"])
        assert returned == ["unused"]  # X returned unchanged

        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "mask_quality_report.json")))
        assert _rel(dim_path, tmp) in _failed_masks(report, "dim_mismatch")
        assert _rel(vocab_path, tmp) in _failed_masks(report, "out_of_vocab_color")
        assert _rel(valid_path, tmp) not in _failed_masks(report, "dim_mismatch")
        assert _rel(valid_path, tmp) not in _failed_masks(report, "out_of_vocab_color")
        assert _rel(valid_path, tmp) not in report["empty"]


def test_empty_mask_flagged():
    """An all-zero mask is flagged as empty (but is not an out-of-vocab failure)."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir, masks_dir = _dirs(tmp)
        cv2.imwrite(os.path.join(images_dir, "99_0_001_empty.png"), np.zeros((32, 32), dtype=np.uint8))
        empty_path = os.path.join(masks_dir, "99_0_001_empty.png")
        cv2.imwrite(empty_path, np.zeros((32, 32), dtype=np.uint8))

        CheckMaskQuality(_make_ctx(tmp)).transform([])
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "mask_quality_report.json")))
        assert _rel(empty_path, tmp) in report["empty"]
        assert _rel(empty_path, tmp) not in _failed_masks(report, "out_of_vocab_color")
