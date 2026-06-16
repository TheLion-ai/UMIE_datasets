"""Unit tests for the StandardizeBitDepth step using small synthetic PNGs (no external data)."""

import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.standardize_bit_depth import StandardizeBitDepth


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


def _write_image(tmp: str, image: np.ndarray) -> str:
    """Write a synthetic image into the UMIE folder layout and return its path."""
    image_dir = os.path.join(tmp, "99_synthetic", "CT", "Images")
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, "99_0_001_x.png")
    cv2.imwrite(image_path, image)
    return image_path


def test_16bit_to_8bit_scaling_no_overflow():
    """16->8 divides by 256: dtype becomes uint8 and values map with no wraparound."""
    with tempfile.TemporaryDirectory() as tmp:
        # Include the max 16-bit value to catch any overflow/wraparound.
        values = np.array([[0, 256, 512, 65535]], dtype=np.uint16)
        image_path = _write_image(tmp, values)

        ctx = _make_ctx(tmp)
        ctx.preprocessing.target_bit_depth = 8
        StandardizeBitDepth(ctx).transform([image_path])

        out = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        assert out.dtype == np.uint8
        # 0//256=0, 256//256=1, 512//256=2, 65535//256=255 (no wraparound to 0).
        np.testing.assert_array_equal(out.reshape(-1), np.array([0, 1, 2, 255], dtype=np.uint8))


def test_no_op_when_target_none():
    """With target_bit_depth unset the image is left byte-identical."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _write_image(tmp, np.array([[1, 2, 3, 4]], dtype=np.uint16))
        before = open(image_path, "rb").read()

        ctx = _make_ctx(tmp)
        assert ctx.preprocessing.target_bit_depth is None
        StandardizeBitDepth(ctx).transform([image_path])

        assert open(image_path, "rb").read() == before
