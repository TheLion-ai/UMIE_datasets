"""Unit tests for the ConvertNii2Nii step using small synthetic NIfTI volumes (no external data)."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from base.selectors.img_selector import BaseImageSelector
from config.dataset_config import DatasetArgs
from constants import OutputMode
from src.steps.convert_nii2nii import ConvertNii2Nii


class _ImgSelector(BaseImageSelector):
    """Selects files whose name contains 'imaging' as images."""

    def _is_image_file(self, path: str) -> bool:
        """Return True for source image files."""
        return "imaging" in os.path.basename(path)


def _make_ctx(tmp: str, output_mode: OutputMode = OutputMode.VOLUMES_3D) -> PipelineContext:
    """Build a minimal PipelineContext for the step (window 40/400, CT modality)."""
    pipeline_args = PipelineArgs(
        window_center=40,
        window_width=400,
        segmentation_prefix="segmentation",
        img_prefix="imaging",
        img_selector=_ImgSelector(),
    )
    identity, dicom, file_selection, output = pipeline_args.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp, output_mode=output_mode),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _save_volume(path: str, data: np.ndarray, spacing: tuple = (1.0, 2.0, 3.0)) -> np.ndarray:
    """Save a volume with a diagonal affine encoding the given voxel spacing; return the affine."""
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(data, affine=affine), path)
    return affine


def test_raises_when_not_3d_mode():
    """The step must refuse to run unless output_mode is VOLUMES_3D."""
    with tempfile.TemporaryDirectory() as tmp:
        step = ConvertNii2Nii(_make_ctx(tmp, output_mode=OutputMode.SLICES_2D))
        with pytest.raises(ValueError):
            step.transform([os.path.join(tmp, "imaging.nii.gz")])


def test_image_volume_windowed_and_geometry_preserved():
    """Image volumes are windowed to 0-255 uint8 while affine and voxel spacing are preserved."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imaging.nii.gz")
        # window_center=40, width=400 -> clip to [-160, 240]
        raw = np.array([-1000.0, -160.0, 40.0, 240.0, 1000.0]).reshape(5, 1, 1)
        affine = _save_volume(path, raw, spacing=(1.0, 2.0, 3.0))

        ConvertNii2Nii(_make_ctx(tmp)).convert_nii2nii(path, is_mask=False)

        out = nib.load(path)
        assert out.get_data_dtype() == np.uint8
        values = out.get_fdata().reshape(-1)
        # clip -> [-160,-160,40,240,240]; shift by 160 -> [0,0,200,400,400]; /(400/255) -> [0,0,127,255,255]
        np.testing.assert_array_equal(values, np.array([0, 0, 127, 255, 255], dtype=float))
        assert values.min() >= 0 and values.max() <= 255
        np.testing.assert_allclose(out.affine, affine)
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.0, 2.0, 3.0))


def test_image_volume_not_windowed_without_params():
    """Without window params the image volume is left untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imaging.nii.gz")
        raw = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1)
        _save_volume(path, raw)
        ctx = _make_ctx(tmp)
        ctx.dicom.window_center = None
        ctx.dicom.window_width = None
        ConvertNii2Nii(ctx).convert_nii2nii(path, is_mask=False)
        np.testing.assert_array_equal(nib.load(path).get_fdata().reshape(-1), raw.reshape(-1))


def test_mask_volume_passthrough_untouched():
    """Mask volumes are never windowed; data, affine and header are preserved exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "segmentation.nii.gz")
        labels = np.array([0, 1, 2, 0, 2], dtype=np.int16).reshape(5, 1, 1)
        affine = _save_volume(path, labels, spacing=(1.5, 1.5, 4.0))

        ConvertNii2Nii(_make_ctx(tmp)).convert_nii2nii(path, is_mask=True)

        out = nib.load(path)
        np.testing.assert_array_equal(out.get_fdata().reshape(-1), labels.reshape(-1).astype(float))
        np.testing.assert_allclose(out.affine, affine)
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.5, 1.5, 4.0))


def test_transform_windows_images_leaves_masks():
    """transform() windows image volumes but leaves mask volumes untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "imaging.nii.gz")
        mask_path = os.path.join(tmp, "segmentation.nii.gz")
        _save_volume(img_path, np.array([-1000.0, 40.0, 1000.0]).reshape(3, 1, 1))
        mask_raw = np.array([0, 1, 2], dtype=np.int16).reshape(3, 1, 1)
        _save_volume(mask_path, mask_raw)

        returned = ConvertNii2Nii(_make_ctx(tmp)).transform([img_path, mask_path])

        assert nib.load(img_path).get_data_dtype() == np.uint8  # image was windowed
        np.testing.assert_array_equal(
            nib.load(mask_path).get_fdata().reshape(-1), mask_raw.reshape(-1).astype(float)
        )  # mask untouched
        assert set(os.path.abspath(p) for p in returned) >= {os.path.abspath(img_path), os.path.abspath(mask_path)}
