"""Unit tests for DetectCorruptImages using small synthetic PNGs (no external data)."""

import json
import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.detect_corrupt_images import DetectCorruptImages


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
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


def _images_dir(tmp: str) -> str:
    """Create and return the synthetic Images folder."""
    path = os.path.join(tmp, "99_synthetic", "CT", "Images")
    os.makedirs(path, exist_ok=True)
    return path


def _rel(path: str, tmp: str) -> str:
    """Return the umie_path (relative to the target/tmp dir) of an output file."""
    return os.path.relpath(path, tmp).replace(os.sep, "/")


def test_valid_blank_and_unreadable_classification():
    """A valid image passes, a garbage file is unreadable, an all-black frame is blank."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)

        valid_path = os.path.join(images_dir, "99_0_001_valid.png")
        rng = np.random.default_rng(0)
        cv2.imwrite(valid_path, rng.integers(0, 256, size=(32, 32), dtype=np.uint8))

        blank_path = os.path.join(images_dir, "99_0_002_blank.png")
        cv2.imwrite(blank_path, np.zeros((32, 32), dtype=np.uint8))

        garbage_path = os.path.join(images_dir, "99_0_003_garbage.png")
        with open(garbage_path, "wb") as handle:
            handle.write(b"not a real png file")

        returned = DetectCorruptImages(_make_ctx(tmp)).transform([valid_path])
        assert returned == [valid_path]  # X returned unchanged

        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "corrupt_images_report.json")))
        assert _rel(valid_path, tmp) not in report["quarantine"]
        assert _rel(blank_path, tmp) in report["blank"]
        assert _rel(garbage_path, tmp) in report["unreadable"]
        assert _rel(blank_path, tmp) in report["quarantine"]
        assert _rel(garbage_path, tmp) in report["quarantine"]


def test_blank_threshold_configurable():
    """A low-variance image is blank only when blank_std_threshold is raised above its std."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        low_var = np.zeros((32, 32), dtype=np.uint8)
        low_var[0, 0] = 5  # tiny variation -> small but non-zero std
        path = os.path.join(images_dir, "99_0_001_lowvar.png")
        cv2.imwrite(path, low_var)

        # default threshold (1.0) -> std is below it, flagged blank
        DetectCorruptImages(_make_ctx(tmp)).transform([path])
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "corrupt_images_report.json")))
        assert _rel(path, tmp) in report["blank"]

        # threshold lowered to 0 -> nothing is blank
        ctx = _make_ctx(tmp)
        ctx.quality.blank_std_threshold = 0.0
        DetectCorruptImages(ctx).transform([path])
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "corrupt_images_report.json")))
        assert _rel(path, tmp) not in report["blank"]


def test_too_small_reported():
    """An image smaller than expected_min_size is reported as too_small."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        rng = np.random.default_rng(1)
        small = os.path.join(images_dir, "99_0_001_small.png")
        cv2.imwrite(small, rng.integers(0, 256, size=(16, 16), dtype=np.uint8))

        ctx = _make_ctx(tmp)
        ctx.quality.expected_min_size = (32, 32)
        DetectCorruptImages(ctx).transform([small])
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "corrupt_images_report.json")))
        assert _rel(small, tmp) in report["too_small"]
        assert _rel(small, tmp) in report["quarantine"]
