"""Tests for 3D geometry preservation and combined 2D+3D output (Theme M: Tasks 40-41)."""

import json
import os
import tempfile

import nibabel as nib
import numpy as np

from base.extractors import BaseStudyIdExtractor
from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from constants import OutputMode
from geometry import (
    build_slice_geometry,
    extract_nifti_geometry,
    geometry_matches,
    reconstruct_volume_from_slices,
    write_slice_geometry_sidecar,
)
from src.steps.store_volumes_alongside import VOLUMES_FOLDER_NAME, StoreVolumesAlongside


def _synthetic_volume() -> "nib.Nifti1Image":
    """A small synthetic volume with a non-trivial affine (anisotropic voxels + translation)."""
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    affine = np.array([[-0.7, 0.0, 0.0, 30.0], [0.0, 0.8, 0.0, -20.0], [0.0, 0.0, 3.0, -50.0], [0.0, 0.0, 0.0, 1.0]])
    return nib.Nifti1Image(data, affine)


# --- Task 40: orientation-preserving slicing + reconstruction -----------------------------------


def test_extract_geometry_fields():
    """Geometry extraction captures affine, codes, voxel sizes, dims and orientation."""
    geometry = extract_nifti_geometry(_synthetic_volume())
    assert geometry["original_dimensions"] == [4, 5, 6]
    assert np.allclose(geometry["voxel_sizes_mm"], [0.7, 0.8, 3.0])
    assert "sform_code" in geometry and "qform_code" in geometry
    assert len(geometry["affine_matrix"]) == 4


def test_build_slice_geometry_records_axis_and_count():
    """The slice-geometry record carries the slicing axis, slice count and pattern."""
    geometry = build_slice_geometry(_synthetic_volume(), slicing_axis=0, png_slice_pattern="vol_{:04d}.png")
    assert geometry["slicing_axis"] == 0
    assert geometry["num_slices"] == 4
    assert geometry["png_slice_pattern"] == "vol_{:04d}.png"


def test_sidecar_round_trips_on_disk():
    """The geometry sidecar is valid JSON and reloads identically."""
    with tempfile.TemporaryDirectory() as tmp:
        geometry = build_slice_geometry(_synthetic_volume(), slicing_axis=0)
        path = write_slice_geometry_sidecar(os.path.join(tmp, "sub", "vol_geometry.json"), geometry)
        assert json.load(open(path)) == geometry


def test_reconstruct_volume_matches_original():
    """Slicing a volume and reconstructing it recovers the shape, affine and (raw) data exactly (Task 40 DOD)."""
    volume = _synthetic_volume()
    data = np.asarray(volume.dataobj)
    geometry = build_slice_geometry(volume, slicing_axis=0)

    # Slice along axis 0 into a list of 2D slices, then reconstruct.
    slices = [data[i, :, :] for i in range(data.shape[0])]
    reconstructed = reconstruct_volume_from_slices(slices, geometry)

    assert list(reconstructed.shape) == geometry["original_dimensions"]
    assert np.allclose(np.asarray(reconstructed.affine), np.asarray(geometry["sform_matrix"]))
    assert np.array_equal(np.asarray(reconstructed.dataobj), data)
    assert geometry_matches(geometry, reconstructed)


# --- Task 41: combined 2D + 3D output mode ------------------------------------------------------


def test_output_mode_has_combined_value():
    """The combined output mode exists and is distinct from the 2D default."""
    assert OutputMode.SLICES_2D_AND_VOLUMES_3D.value == "slices_2d_and_volumes_3d"
    assert OutputMode.SLICES_2D_AND_VOLUMES_3D != OutputMode.SLICES_2D


class _StudyIdExtractor(BaseStudyIdExtractor):
    """Derive the study id from the source case folder (e.g. ``.../case_3/3.nii.gz`` -> ``3``)."""

    def _extract(self, img_path: str) -> str:
        return self._extract_parent_dir(img_path, parent_dir_level=1, include_path=False).split("_")[-1]


def _ctx(tmp: str, output_mode: OutputMode) -> PipelineContext:
    """Build a PipelineContext rooted at ``tmp`` with the given output mode."""
    pa = PipelineArgs(study_id_extractor=_StudyIdExtractor())
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=os.path.join(tmp, "src"), target_path=tmp, output_mode=output_mode),
        dataset=DatasetArgs(dataset_uid="07", dataset_name="lits", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_source_volume(src_dir: str, name: str) -> str:
    """Write a synthetic source volume and return its path."""
    os.makedirs(src_dir, exist_ok=True)
    path = os.path.join(src_dir, name)
    nib.save(_synthetic_volume(), path)
    return path


def test_store_volumes_alongside_in_combined_mode():
    """Combined mode copies the source volume into <modality>/Volumes/ under a UMIE id, geometry intact."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(tmp, OutputMode.SLICES_2D_AND_VOLUMES_3D)
        src_dir = os.path.join(ctx.paths.source_path, "case_3")
        _write_source_volume(src_dir, "3.nii.gz")

        StoreVolumesAlongside(ctx).transform(["unchanged"])

        volumes_dir = os.path.join(tmp, "07_lits", "CT", VOLUMES_FOLDER_NAME)
        produced = os.listdir(volumes_dir)
        assert produced == ["07_0_3_3.nii.gz"]  # {uid}_{modality}_{study}_{img}.nii.gz
        # the copied volume preserves the original geometry
        copied = nib.load(os.path.join(volumes_dir, produced[0]))
        assert np.allclose(np.asarray(copied.affine), np.asarray(_synthetic_volume().affine))


def test_store_volumes_alongside_is_noop_in_2d_mode():
    """In the default 2D mode the step writes no Volumes/ folder (output stays byte-identical)."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(tmp, OutputMode.SLICES_2D)
        _write_source_volume(os.path.join(ctx.paths.source_path, "case_3"), "3.nii.gz")

        StoreVolumesAlongside(ctx).transform(["unchanged"])

        assert not os.path.exists(os.path.join(tmp, "07_lits", "CT", VOLUMES_FOLDER_NAME))
