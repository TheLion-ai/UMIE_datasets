"""Unit tests for conditional 2D/3D step selection in pipelines (synthetic data)."""

import glob
import os
import tempfile

import cv2
import nibabel as nib
import numpy as np
import pytest

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from base.selectors.img_selector import BaseImageSelector
from config.dataset_config import DatasetArgs
from constants import OutputMode
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.coca import COCAPipeline
from src.pipelines.kits23 import KITS23Pipeline
from src.steps.convert_nii2nii import ConvertNii2Nii
from src.steps.convert_nii2png import ConvertNii2Png


def _step_names(pipeline) -> list:
    """Return the ordered step names of a built sklearn pipeline."""
    return [name for name, _ in pipeline.pipeline.steps]


def test_nii_pipeline_swaps_to_3d_step():
    """KITS-23 swaps convert_nii2png -> convert_nii2nii in 3D mode, keeps it in 2D."""
    names_2d = _step_names(KITS23Pipeline(path_args=PathArgs(source_path="", target_path="/d")))
    assert "convert_nii2png" in names_2d and "convert_nii2nii" not in names_2d

    names_3d = _step_names(
        KITS23Pipeline(path_args=PathArgs(source_path="", target_path="/d", output_mode=OutputMode.VOLUMES_3D))
    )
    assert "convert_nii2nii" in names_3d and "convert_nii2png" not in names_3d


def test_dcm_pipeline_swaps_to_3d_step():
    """COCA swaps convert_dcm2png -> convert_dcm2nii in 3D mode."""
    names_3d = _step_names(
        COCAPipeline(path_args=PathArgs(source_path="", target_path="/d", output_mode=OutputMode.VOLUMES_3D))
    )
    assert "convert_dcm2nii" in names_3d and "convert_dcm2png" not in names_3d


def test_inherently_2d_dataset_warns_and_falls_back():
    """An inherently-2D dataset requesting 3D warns and falls back to 2D (no crash)."""
    with pytest.warns(UserWarning, match="falls back to 2D"):
        pipe = AlzheimersPipeline(
            path_args=PathArgs(source_path="", target_path="/d", output_mode=OutputMode.VOLUMES_3D)
        )
    assert pipe.ctx.paths.output_mode == OutputMode.SLICES_2D
    # pipeline still builds and contains no 3D conversion step
    names = _step_names(pipe)
    assert "convert_nii2nii" not in names and "convert_dcm2nii" not in names


class _ImgSelector(BaseImageSelector):
    """Selects files whose name contains 'imaging'."""

    def _is_image_file(self, path: str) -> bool:
        """Return True for source image files."""
        return "imaging" in os.path.basename(path)


def _ctx(tmp: str, output_mode: OutputMode) -> PipelineContext:
    """Build a context with windowing + zfill for the conversion steps."""
    pa = PipelineArgs(
        window_center=40,
        window_width=400,
        zfill=2,
        segmentation_prefix="segmentation",
        img_prefix="imaging",
        img_selector=_ImgSelector(),
    )
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp, output_mode=output_mode),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def test_roundtrip_2d_slices_match_3d_volume():
    """Reassembling the 2D PNG slices reproduces the 3D NIfTI volume for a windowed case."""
    with tempfile.TemporaryDirectory() as tmp:
        # Build a volume where EVERY axis-0 slice spans the window extremes, so per-slice
        # (2D) and whole-volume (3D) windowing coincide and the round-trip is exact.
        nslices, rows, cols = 3, 4, 5
        data = np.full((nslices, rows, cols), 40.0)
        data[:, 0, 0] = -1000.0
        data[:, 0, 1] = 1000.0
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

        path_2d = os.path.join(tmp, "imaging2d.nii.gz")
        path_3d = os.path.join(tmp, "imaging3d.nii.gz")
        nib.save(nib.Nifti1Image(data.copy(), affine=affine), path_2d)
        nib.save(nib.Nifti1Image(data.copy(), affine=affine), path_3d)

        # 2D path: slice to PNGs
        ConvertNii2Png(_ctx(tmp, OutputMode.SLICES_2D)).convert_nii2png(path_2d)
        # 3D path: window the whole volume in place
        ConvertNii2Nii(_ctx(tmp, OutputMode.VOLUMES_3D)).convert_nii2nii(path_3d, is_mask=False)

        png_paths = sorted(glob.glob(os.path.join(tmp, "imaging2d_*.png")))
        assert len(png_paths) == nslices
        reassembled = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in png_paths], axis=0)

        volume_3d = nib.load(path_3d).get_fdata().astype(np.uint8)
        assert reassembled.shape == volume_3d.shape == (nslices, rows, cols)
        np.testing.assert_allclose(reassembled, volume_3d, atol=1)
