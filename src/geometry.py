"""Orientation-preserving 2D slicing + 3D reconstruction (Theme M: Task 40).

PNG files cannot store the 3D spatial orientation a NIfTI header carries (sform/qform affines,
voxel spacing, codes). When a volume is sliced to PNGs we therefore persist that geometry in a
**sidecar** (per ``Nifti to png.md``) so the 2D slices can be mapped back into 3D physical space,
and provide :func:`reconstruct_volume_from_slices` to stack the slices back into a volume.

This is purely additive: the PNG pixels and filenames are untouched (2D default stays
byte-identical); the geometry lives in a separate JSON sidecar (v1) and also feeds the v2
``nifti_files`` / ``png_representation`` blocks (Theme K).
"""

from __future__ import annotations

import json
import os
from typing import Any, Union

import nibabel as nib
import numpy as np

NiftiLike = Union[str, "nib.Nifti1Image"]


def _load(nii: NiftiLike) -> "nib.Nifti1Image":
    """Return a NIfTI image, loading from a path if needed."""
    return nib.load(nii) if isinstance(nii, str) else nii  # type: ignore[return-value]


def extract_nifti_geometry(nii: NiftiLike) -> dict:
    """Extract the orientation metadata needed to map slices back to 3D space (Task 40).

    Args:
        nii: A NIfTI image or a path to a ``.nii.gz`` file.

    Returns:
        dict: affine, sform/qform matrices and codes, voxel sizes, dimensions and orientation.
    """
    img = _load(nii)
    header = img.header
    affine = np.asarray(img.affine)
    sform = np.asarray(img.get_sform())  # type: ignore[no-untyped-call]
    qform = np.asarray(img.get_qform())  # type: ignore[no-untyped-call]
    return {
        "affine_matrix": affine.tolist(),
        "sform_matrix": sform.tolist(),
        "sform_code": int(header["sform_code"]),  # type: ignore[index]
        "qform_matrix": qform.tolist(),
        "qform_code": int(header["qform_code"]),  # type: ignore[index]
        "voxel_sizes_mm": [float(z) for z in header.get_zooms()[:3]],  # type: ignore[no-untyped-call]
        "original_dimensions": [int(d) for d in img.shape],
        "orientation": "".join(nib.aff2axcodes(affine)),  # type: ignore[no-untyped-call]
    }


def build_slice_geometry(
    nii: NiftiLike,
    *,
    slicing_axis: int = 0,
    png_slice_pattern: str = "{:04d}.png",
    original_nifti_filename: str | None = None,
) -> dict:
    """Build the full slice-geometry record for one volume (geometry + how it was sliced).

    Args:
        nii: The source NIfTI image or path.
        slicing_axis: Axis the volume is sliced along (0/1/2). ``ConvertNii2Png`` slices axis 0.
        png_slice_pattern: Filename pattern mapping a slice index to its PNG.
        original_nifti_filename: Original filename, for reference.

    Returns:
        dict: ``extract_nifti_geometry`` plus ``slicing_axis``, ``num_slices`` and the slice pattern.
    """
    geometry = extract_nifti_geometry(nii)
    geometry.update(
        {
            "original_nifti_filename": original_nifti_filename,
            "slicing_axis": slicing_axis,
            "num_slices": geometry["original_dimensions"][slicing_axis],
            "png_slice_pattern": png_slice_pattern,
        }
    )
    return geometry


def write_slice_geometry_sidecar(sidecar_path: str, geometry: dict) -> str:
    """Write the slice-geometry record to ``sidecar_path`` (creating parent dirs). Returns the path."""
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    with open(sidecar_path, "w") as handle:
        json.dump(geometry, handle, indent=2)
    return sidecar_path


def reconstruct_volume_from_slices(slices: Union[list, np.ndarray], geometry: dict) -> "nib.Nifti1Image":
    """Stack 2D slices back into a 3D NIfTI volume using the stored geometry (Task 40).

    Args:
        slices: Slices in acquisition order - a list of 2D arrays, or a 3D array already stacked
            along ``slicing_axis``.
        geometry: A record from :func:`build_slice_geometry` (provides the slicing axis + affine).

    Returns:
        nib.Nifti1Image: The reconstructed volume. Its shape and affine match the original within
        tolerance; intensities match exactly for raw slices (PNG windowing/quantisation is lossy).
    """
    axis = geometry.get("slicing_axis", 0)
    volume = np.asarray(slices) if isinstance(slices, np.ndarray) else np.stack(list(slices), axis=axis)
    affine = np.asarray(geometry.get("sform_matrix") or geometry.get("affine_matrix"))
    return nib.Nifti1Image(volume, affine)  # type: ignore[no-untyped-call]


def geometry_matches(geometry: dict, reconstructed: "nib.Nifti1Image", atol: float = 1e-4) -> bool:
    """Return True if a reconstructed volume's shape and affine match the stored geometry."""
    same_shape = list(reconstructed.shape) == list(geometry["original_dimensions"])
    expected_affine = np.asarray(geometry.get("sform_matrix") or geometry.get("affine_matrix"))
    same_affine: bool = bool(np.allclose(np.asarray(reconstructed.affine), expected_affine, atol=atol))
    return same_shape and same_affine
