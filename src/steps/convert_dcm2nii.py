"""Reconstruct NIfTI volumes from DICOM series for 3D output."""

import glob
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import pydicom
import pydicom.pixel_data_handlers.util as ddh
from pydicom import dcmread
from tqdm import tqdm

from base.step import BaseStep
from constants import OutputMode


class ConvertDcm2Nii(BaseStep):
    """Reconstruct ``.nii.gz`` volumes from DICOM series for ``VOLUMES_3D`` output.

    DICOM files are grouped by ``SeriesInstanceUID``, slices are ordered along the slice
    normal (projecting ``ImagePositionPatient``, falling back to ``InstanceNumber``), and the
    affine is built from the DICOM geometry tags (``ImageOrientationPatient``,
    ``ImagePositionPatient``, ``PixelSpacing`` and the inter-slice spacing). One ``.nii.gz``
    is written per series.

    Backend decision: geometry is built directly with ``pydicom`` (already a project
    dependency) rather than adding ``dicom2nifti`` / ``dcm2niix``. This keeps the step
    dependency-free and unit-testable, and the affine follows the DICOM PS3.3 C.7.6.2
    image-plane formula so it can be validated against a known reference geometry.
    """

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Group DICOM files into series and reconstruct one ``.nii.gz`` per series.

        Args:
            X (list): List of paths (DICOM files are selected from it; non-``.dcm`` entries
                are ignored). If empty, all ``.dcm`` files under ``source_path`` are used.

        Returns:
            list: Paths to the reconstructed ``.nii.gz`` volumes.

        Raises:
            ValueError: If the pipeline is not running in ``VOLUMES_3D`` output mode.
        """
        if self.output_mode != OutputMode.VOLUMES_3D:
            raise ValueError(
                f"ConvertDcm2Nii can only run in {OutputMode.VOLUMES_3D} mode, "
                f"but output_mode is {self.output_mode}. Use ConvertDcm2Png for 2D slices."
            )
        print("Reconstructing nii volumes from dicom series...")
        dcm_paths = [p for p in X if p.endswith(".dcm")]
        if not dcm_paths:
            dcm_paths = glob.glob(os.path.join(self.source_path, "**/*.dcm"), recursive=True)
        if not dcm_paths:
            raise ValueError("No DICOM files provided.")

        series = self._group_by_series(dcm_paths)
        out_paths = []
        for series_uid, files in tqdm(series.items()):
            out_path = self._convert_series(series_uid, files)
            if out_path is not None:
                out_paths.append(out_path)
        return out_paths

    def _group_by_series(self, dcm_paths: list) -> dict:
        """Group DICOM file paths by their ``SeriesInstanceUID``.

        Args:
            dcm_paths (list): DICOM file paths.

        Returns:
            dict: Mapping of SeriesInstanceUID to the list of its file paths.
        """
        series: dict = defaultdict(list)
        for path in dcm_paths:
            ds = dcmread(path, stop_before_pixels=True)
            series[str(ds.SeriesInstanceUID)].append(path)
        return series

    def _slice_sort_key(self, ds: pydicom.dataset.FileDataset, normal: np.ndarray) -> float:
        """Return the position of a slice along the slice normal (fallback: InstanceNumber)."""
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(np.dot(np.array(ipp, dtype=float), normal))
        return float(getattr(ds, "InstanceNumber", 0))

    def _build_affine(self, slices: list) -> np.ndarray:
        """Build the voxel-to-patient affine from the ordered slices' geometry tags.

        Args:
            slices (list): DICOM datasets ordered along the slice normal.

        Returns:
            np.ndarray: 4x4 affine mapping voxel (col, row, slice) indices to patient space.
        """
        first = slices[0]
        iop = np.array(first.ImageOrientationPatient, dtype=float)
        row_cosine, col_cosine = iop[0:3], iop[3:6]  # X (along columns), Y (along rows)
        pixel_spacing = [float(v) for v in first.PixelSpacing]  # [row spacing, column spacing]
        ipp_first = np.array(first.ImagePositionPatient, dtype=float)

        affine = np.eye(4)
        affine[:3, 0] = row_cosine * pixel_spacing[1]  # i index runs along columns
        affine[:3, 1] = col_cosine * pixel_spacing[0]  # j index runs along rows
        if len(slices) > 1:
            ipp_last = np.array(slices[-1].ImagePositionPatient, dtype=float)
            affine[:3, 2] = (ipp_last - ipp_first) / (len(slices) - 1)
        else:
            normal = np.cross(row_cosine, col_cosine)
            affine[:3, 2] = normal * float(getattr(first, "SliceThickness", 1.0) or 1.0)
        affine[:3, 3] = ipp_first
        return affine

    def _convert_series(self, series_uid: str, files: list) -> str:
        """Reconstruct and save a single series as ``.nii.gz``.

        Args:
            series_uid (str): The SeriesInstanceUID.
            files (list): DICOM file paths belonging to the series.

        Returns:
            str: Path to the written ``.nii.gz`` volume.
        """
        datasets = [dcmread(path) for path in files]
        iop = np.array(datasets[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[0:3], iop[3:6])
        datasets.sort(key=lambda ds: self._slice_sort_key(ds, normal))

        # Stack slices into a (columns, rows, slices) volume so it matches the affine below.
        volume = np.stack([ddh.apply_modality_lut(ds.pixel_array, ds).T for ds in datasets], axis=-1)
        if self.window_center is not None and self.window_width is not None:
            volume = self._apply_window(volume)

        affine = self._build_affine(datasets)
        out_path = os.path.join(os.path.dirname(files[0]), f"{series_uid}.nii.gz")
        nib.save(nib.Nifti1Image(volume, affine=affine), out_path)  # type: ignore[no-untyped-call]  # nibabel stub gap
        return out_path

    def _apply_window(self, pixel_data: np.ndarray) -> np.ndarray:
        """Clip a volume to the configured window and rescale it to the 0-255 range.

        Args:
            pixel_data (np.ndarray): Raw volume intensities.

        Returns:
            np.ndarray: Windowed volume as ``uint8``.
        """
        pixel_data = np.clip(
            pixel_data,
            self.window_center - self.window_width / 2,
            self.window_center + self.window_width / 2,
        )
        min_val = np.min(pixel_data)
        min_val = -1000 if min_val < -1000 else min_val
        pixel_data = pixel_data - min_val
        ratio = np.max(pixel_data) / 255
        if ratio == 0:
            ratio = 1
        pixel_data = np.divide(pixel_data, ratio)
        return pixel_data.astype(np.uint8)
