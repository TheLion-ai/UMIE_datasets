"""Unit tests for the NormalizeSpacing step using small synthetic NIfTI volumes (no external data)."""

import os
import tempfile

import nibabel as nib
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from constants import OutputMode
from src.steps.normalize_spacing import NormalizeSpacing


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a 3D-mode PipelineContext rooted at ``tmp`` (synthetic dataset uid 99)."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp, output_mode=OutputMode.VOLUMES_3D),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _save_volume(path: str, data: np.ndarray, spacing: tuple) -> None:
    """Save ``data`` with a diagonal affine encoding ``spacing`` (one value per spatial axis)."""
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=affine), path)


def test_image_volume_resampled_to_target_spacing():
    """A (1,1,3) image -> target (1,1,1): header zooms become target and shape scales by 3 on z."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "99_synthetic", "CT", "Images", "99_0_001_x.nii.gz")
        data = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        _save_volume(image_path, data, spacing=(1.0, 1.0, 3.0))

        ctx = _make_ctx(tmp)
        ctx.preprocessing.target_spacing_mm = (1.0, 1.0, 1.0)
        NormalizeSpacing(ctx).transform([image_path])

        out = nib.load(image_path)
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.0, 1.0, 1.0), atol=1e-5)
        # z axis spacing 3 -> 1 means roughly 3x more slices; x/y unchanged.
        assert out.shape[0] == 4 and out.shape[1] == 4
        assert out.shape[2] == 12


def test_mask_volume_nearest_introduces_no_new_labels():
    """A mask resampled with order=0 keeps its discrete label set (no interpolation)."""
    with tempfile.TemporaryDirectory() as tmp:
        mask_path = os.path.join(tmp, "99_synthetic", "CT", "Masks", "99_0_001_x.nii.gz")
        labels = np.zeros((4, 4, 4), dtype=np.int16)
        labels[1:3, 1:3, 0:2] = 2
        labels[0, 0, 2:4] = 5
        source_values = set(np.unique(labels).tolist())
        _save_volume(mask_path, labels, spacing=(1.0, 1.0, 3.0))

        ctx = _make_ctx(tmp)
        ctx.preprocessing.target_spacing_mm = (1.0, 1.0, 1.0)
        NormalizeSpacing(ctx).transform([])

        out = nib.load(mask_path)
        out_values = set(np.unique(out.get_fdata().astype(np.int16)).tolist())
        assert out_values.issubset(source_values)
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.0, 1.0, 1.0), atol=1e-5)


def test_no_op_when_target_spacing_none():
    """With target_spacing_mm unset the volume is left untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, "99_synthetic", "CT", "Images", "99_0_001_x.nii.gz")
        data = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
        _save_volume(image_path, data, spacing=(1.0, 1.0, 3.0))

        ctx = _make_ctx(tmp)
        assert ctx.preprocessing.target_spacing_mm is None
        NormalizeSpacing(ctx).transform([image_path])

        out = nib.load(image_path)
        assert out.shape == (2, 2, 2)
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.0, 1.0, 3.0), atol=1e-5)
