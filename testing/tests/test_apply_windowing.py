"""Unit tests for the ApplyWindowing step using small synthetic PNGs (no external data)."""

import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.apply_windowing import WINDOW_PRESETS, ApplyWindowing


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext rooted at ``tmp`` (synthetic CT dataset uid 99)."""
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


def test_preset_windowing_maps_known_values():
    """A brain preset (40, 80) -> window [0, 80]; values map linearly to 0-255 uint8."""
    with tempfile.TemporaryDirectory() as tmp:
        # window [0,80]: 0->0, 40->127(.5 rounded), 80->255, and out-of-window values clip.
        values = np.array([[0, 40, 80, 200]], dtype=np.uint16)
        image_path, mask_path = _write_dataset(tmp, values, np.array([[0, 1, 2, 0]], dtype=np.uint8))

        ctx = _make_ctx(tmp)
        ctx.preprocessing.window_preset = "brain"
        ApplyWindowing(ctx).transform([image_path])

        out = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        assert out.dtype == np.uint8
        center, width = WINDOW_PRESETS["brain"]
        lower = center - width / 2
        expected = np.clip(np.array([0, 40, 80, 200], float), lower, lower + width)
        expected = ((expected - lower) * (255.0 / width)).round().astype(np.uint8)
        np.testing.assert_array_equal(out.reshape(-1), expected)
        assert out.min() >= 0 and out.max() <= 255


def test_center_width_fallback_used_when_no_preset():
    """Without a preset, window_center/window_width from DICOM config drive the mapping."""
    with tempfile.TemporaryDirectory() as tmp:
        values = np.array([[0, 50, 100, 150]], dtype=np.uint16)
        image_path, _ = _write_dataset(tmp, values, np.zeros((1, 4), dtype=np.uint8))

        ctx = _make_ctx(tmp)
        ctx.dicom.window_center = 50  # window [0, 100]
        ctx.dicom.window_width = 100
        ApplyWindowing(ctx).transform([image_path])

        out = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        assert out.dtype == np.uint8
        expected = np.clip(np.array([0, 50, 100, 150], float), 0, 100)
        expected = (expected * (255.0 / 100)).round().astype(np.uint8)
        np.testing.assert_array_equal(out.reshape(-1), expected)


def test_no_op_when_no_window_configured():
    """With neither preset nor center/width, the image is left byte-identical."""
    with tempfile.TemporaryDirectory() as tmp:
        values = np.array([[10, 20, 30, 40]], dtype=np.uint16)
        image_path, _ = _write_dataset(tmp, values, np.zeros((1, 4), dtype=np.uint8))
        before = open(image_path, "rb").read()

        ApplyWindowing(_make_ctx(tmp)).transform([image_path])

        assert open(image_path, "rb").read() == before


def test_masks_untouched():
    """Windowing must never modify the paired mask."""
    with tempfile.TemporaryDirectory() as tmp:
        image = np.array([[0, 40, 80, 200]], dtype=np.uint16)
        mask = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        image_path, mask_path = _write_dataset(tmp, image, mask)
        mask_before = open(mask_path, "rb").read()

        ctx = _make_ctx(tmp)
        ctx.preprocessing.window_preset = "brain"
        ApplyWindowing(ctx).transform([image_path])

        assert open(mask_path, "rb").read() == mask_before
