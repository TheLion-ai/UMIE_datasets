"""Unit tests for the AutocropBorders step using small synthetic PNGs (no external data)."""

import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.autocrop_borders import AutocropBorders


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


def test_crops_to_content_bbox_and_aligns_mask():
    """A black-bordered content block is cropped to its bbox; the mask gets the same crop."""
    with tempfile.TemporaryDirectory() as tmp:
        # 20x20 black frame with a 8x6 content block at rows 5..13, cols 4..10.
        image = np.zeros((20, 20), dtype=np.uint8)
        image[5:13, 4:10] = 200
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[6:10, 5:8] = 3
        image_path, mask_path = _write_dataset(tmp, image, mask)

        ctx = _make_ctx(tmp)
        ctx.preprocessing.autocrop_enabled = True
        AutocropBorders(ctx).transform([image_path])

        out_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        out_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert out_image.shape == (8, 6)  # cropped exactly to content bbox
        assert out_mask.shape == (8, 6)  # mask cropped to the identical region
        assert out_image.min() == 200  # only content remains
        # Mask label survives at the same relative position (mask rows 6..9 -> cropped rows 1..4).
        assert np.array_equal(np.unique(out_mask), np.array([0, 3], dtype=np.uint8))


def test_no_op_when_disabled():
    """With autocrop_enabled False the image is left byte-identical."""
    with tempfile.TemporaryDirectory() as tmp:
        image = np.zeros((20, 20), dtype=np.uint8)
        image[5:13, 4:10] = 200
        image_path, _ = _write_dataset(tmp, image, np.zeros((20, 20), dtype=np.uint8))
        before = open(image_path, "rb").read()

        ctx = _make_ctx(tmp)
        assert ctx.preprocessing.autocrop_enabled is False
        AutocropBorders(ctx).transform([image_path])

        assert open(image_path, "rb").read() == before
