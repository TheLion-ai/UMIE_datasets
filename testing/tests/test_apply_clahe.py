"""Unit tests for the ApplyClahe step using small synthetic PNGs (no external data)."""

import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.apply_clahe import ApplyClahe


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext rooted at ``tmp`` (synthetic dataset uid 99)."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
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


def _low_contrast_image() -> np.ndarray:
    """Return a low-contrast 8-bit gradient image CLAHE will visibly alter."""
    base = np.linspace(100, 140, 64, dtype=np.uint8)
    return np.tile(base, (64, 1))


def test_disabled_is_byte_identical_no_op():
    """When clahe_enabled is False the output file is byte-identical (no write)."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _write_image(tmp, _low_contrast_image())
        before = open(image_path, "rb").read()

        ctx = _make_ctx(tmp)
        assert ctx.preprocessing.clahe_enabled is False
        ApplyClahe(ctx).transform([image_path])

        assert open(image_path, "rb").read() == before


def test_enabled_produces_valid_png_and_changes_low_contrast():
    """When enabled, output stays a readable uint8 PNG and differs for a low-contrast input."""
    with tempfile.TemporaryDirectory() as tmp:
        original = _low_contrast_image()
        image_path = _write_image(tmp, original)

        ctx = _make_ctx(tmp)
        ctx.preprocessing.clahe_enabled = True
        ApplyClahe(ctx).transform([image_path])

        out = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        assert out is not None  # still a valid, readable PNG
        assert out.dtype == np.uint8
        assert out.shape == original.shape
        assert not np.array_equal(out, original)  # contrast was enhanced
