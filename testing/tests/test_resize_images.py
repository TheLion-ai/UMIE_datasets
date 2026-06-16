"""Unit tests for the ResizeImages step using small synthetic PNGs (no external data)."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.resize_images import ResizeImages


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext rooted at ``tmp`` (synthetic dataset uid 99)."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_dataset(tmp: str, image: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
    """Write a synthetic image+mask pair into the UMIE folder layout; return their paths."""
    image_dir = os.path.join(tmp, "99_synthetic", "CT", "Images")
    mask_dir = os.path.join(tmp, "99_synthetic", "CT", "Masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    image_path = os.path.join(image_dir, "99_0_001_x.png")
    mask_path = os.path.join(mask_dir, "99_0_001_x.png")
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)
    return image_path, mask_path


@pytest.mark.parametrize("strategy", ["pad", "crop", "letterbox", "stretch"])
def test_strategy_outputs_target_size_and_preserves_mask_labels(strategy: str):
    """Each strategy yields target_size for image and mask; mask gains no new label values."""
    with tempfile.TemporaryDirectory() as tmp:
        # Non-square 40x80 input (h x w); mask carries discrete labels.
        image = np.random.randint(0, 256, size=(40, 80), dtype=np.uint8)
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[10:30, 20:60] = 4
        mask[0:5, 0:5] = 7
        source_labels = set(np.unique(mask).tolist())
        image_path, mask_path = _write_dataset(tmp, image, mask)

        ctx = _make_ctx(tmp)
        ctx.preprocessing.target_size = (32, 32)
        ctx.preprocessing.resize_strategy = strategy
        ResizeImages(ctx).transform([image_path])

        out_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        out_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert out_image.shape == (32, 32)
        assert out_mask.shape == (32, 32)
        out_labels = set(np.unique(out_mask).tolist())
        assert out_labels.issubset(source_labels)


def test_no_op_when_target_size_none():
    """With target_size unset the image is left byte-identical."""
    with tempfile.TemporaryDirectory() as tmp:
        image = np.random.randint(0, 256, size=(40, 80), dtype=np.uint8)
        image_path, _ = _write_dataset(tmp, image, np.zeros((40, 80), dtype=np.uint8))
        before = open(image_path, "rb").read()

        ctx = _make_ctx(tmp)
        assert ctx.preprocessing.target_size is None
        ResizeImages(ctx).transform([image_path])

        assert open(image_path, "rb").read() == before
