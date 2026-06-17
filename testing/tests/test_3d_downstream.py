"""Unit tests for 3D-aware downstream steps (synthetic data, no external dependencies)."""

import os
import tempfile

import nibabel as nib
import numpy as np

from base.pipeline import PathArgs
from config.dataset_config import kits23
from constants import OutputMode
from src.pipelines.kits23 import KITS23Pipeline
from src.steps.add_umie_ids import AddUmieIds
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.recolor_masks import RecolorMasks
from src.steps.validate_data import ValidateData


def _pipeline(tmp: str, output_mode: OutputMode) -> KITS23Pipeline:
    """Build a KITS23 pipeline (no source) with the given output mode."""
    return KITS23Pipeline(path_args=PathArgs(source_path="", target_path=tmp, output_mode=output_mode))


def _save_volume(path: str, data: np.ndarray, spacing=(0.8, 0.8, 3.0)) -> None:
    """Save a volume with a diagonal affine encoding the given voxel spacing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(data, affine=affine), path)


def test_get_umie_id_preserves_nii_gz_in_3d_and_png_in_2d():
    """UMIE ids keep .nii.gz in 3D mode and stay .png in 2D mode (id structure unchanged)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = "/x/case_00123/imaging.nii.gz"
        step_2d = RecolorMasks(_pipeline(tmp, OutputMode.SLICES_2D).ctx)
        step_3d = RecolorMasks(_pipeline(tmp, OutputMode.VOLUMES_3D).ctx)
        # 2D behavior is unchanged (regression): extension forced to .png
        assert step_2d.get_umie_id(src) == "00_0_00123_imaging.nii.png"
        # 3D preserves the volumetric extension; uid/modality/study structure identical
        assert step_3d.get_umie_id(src) == "00_0_00123_imaging.nii.gz"


def test_recolor_masks_3d_remaps_full_volume():
    """RecolorMasks remaps voxel values across the whole NIfTI volume in 3D mode."""
    with tempfile.TemporaryDirectory() as tmp:
        mask_path = os.path.join(tmp, "00_kits23", "CT", "Masks", "00_0_001_seg.nii.gz")
        _save_volume(mask_path, np.array([0, 1, 2, 3], dtype=np.int16).reshape(4, 1, 1), spacing=(1.0, 1.0, 2.0))

        RecolorMasks(_pipeline(tmp, OutputMode.VOLUMES_3D).ctx).transform(["dummy"])

        out = nib.load(mask_path)
        values = out.get_fdata().reshape(-1)
        expected = [
            0,
            kits23.masks["Kidney"].target_color,
            kits23.masks["Neoplasm"].target_color,
            kits23.masks["RenalCyst"].target_color,
        ]
        np.testing.assert_array_equal(values, np.array(expected, dtype=float))
        assert out.get_data_dtype() == np.uint8
        np.testing.assert_allclose(out.header.get_zooms()[:3], (1.0, 1.0, 2.0))  # geometry preserved


def test_add_umie_ids_jsonl_volume_metadata_3d_only():
    """3D JSONL gains output_mode + volume_metadata; 2D JSONL has neither (additive only)."""
    with tempfile.TemporaryDirectory() as tmp:
        umie_path = os.path.join(tmp, "00_kits23", "CT", "Images", "00_0_001_imaging.nii.gz")
        _save_volume(umie_path, np.zeros((5, 6, 7), dtype=np.int16), spacing=(0.8, 0.8, 3.0))

        step_3d = AddUmieIds(_pipeline(tmp, OutputMode.VOLUMES_3D).ctx)
        step_3d.new_json = []
        step_3d._update_json(umie_path, "")
        rec = step_3d.new_json[0]
        assert rec["output_mode"] == "volumes_3d"
        vm = rec["volume_metadata"]
        assert vm["shape"] == [5, 6, 7]
        np.testing.assert_allclose(vm["voxel_spacing_mm"], [0.8, 0.8, 3.0], rtol=1e-5)
        assert isinstance(vm["orientation"], str) and len(vm["orientation"]) == 3
        assert len(vm["affine"]) == 4 and len(vm["affine"][0]) == 4
        # existing fields untouched
        assert rec["umie_id"] == "00_0_001_imaging.nii.gz" and rec["dataset_uid"] == "00"

        # 2D: no output_mode / volume_metadata keys (byte-identical schema)
        step_2d = AddUmieIds(_pipeline(tmp, OutputMode.SLICES_2D).ctx)
        step_2d.new_json = []
        step_2d._update_json(os.path.join(tmp, "00_kits23", "CT", "Images", "00_0_001_x.png"), "")
        assert "output_mode" not in step_2d.new_json[0]
        assert "volume_metadata" not in step_2d.new_json[0]


def test_validate_data_accepts_nii_gz():
    """ValidateData._is_readable accepts valid .nii.gz volumes and rejects missing ones."""
    with tempfile.TemporaryDirectory() as tmp:
        vol = os.path.join(tmp, "vol.nii.gz")
        _save_volume(vol, np.ones((2, 2, 2), dtype=np.int16))
        step = ValidateData(_pipeline(tmp, OutputMode.VOLUMES_3D).ctx)
        assert step._is_readable(vol) is True
        assert step._is_readable(os.path.join(tmp, "missing.nii.gz")) is False


def test_delete_temp_png_is_noop_in_3d():
    """DeleteTempPng must not delete source PNGs when in 3D mode."""
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "src", "slice.png")
        os.makedirs(os.path.dirname(png))
        with open(png, "wb") as f:
            f.write(b"not-really-a-png")
        ctx = _pipeline(tmp, OutputMode.VOLUMES_3D).ctx
        ctx.paths.source_path = os.path.join(tmp, "src")
        DeleteTempPng(ctx).transform(["dummy"])
        assert os.path.exists(png)  # untouched in 3D
