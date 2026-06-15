"""Standardize NIfTI volumes for 3D output (instead of slicing them to PNG)."""

import glob
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

from base.step import BaseStep
from constants import OutputMode


class ConvertNii2Nii(BaseStep):
    """Standardize/copy NIfTI volumes for ``VOLUMES_3D`` output.

    This is the 3D counterpart of :class:`ConvertNii2Png`. Instead of writing one PNG per
    slice it keeps each volume as a ``.nii.gz``: image volumes are optionally windowed
    (using ``DicomConfig`` window parameters), while mask volumes are passed through
    untouched. The affine, voxel spacing and header are preserved.
    """

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Standardize NIfTI volumes in place for 3D output.

        Args:
            X (list): List of paths to the source NIfTI files.

        Returns:
            list: List of paths to the standardized NIfTI volumes.

        Raises:
            ValueError: If the pipeline is not running in ``VOLUMES_3D`` output mode, or if
                no files are provided.
        """
        if self.output_mode != OutputMode.VOLUMES_3D:
            raise ValueError(
                f"ConvertNii2Nii can only run in {OutputMode.VOLUMES_3D} mode, "
                f"but output_mode is {self.output_mode}. Use ConvertNii2Png for 2D slices."
            )
        print("Standardizing nii volumes...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            if not img_path.endswith(".nii.gz"):
                continue
            is_mask = bool(self.segmentation_prefix) and self.segmentation_prefix in os.path.basename(img_path)
            is_image = self.img_selector(img_path) if self.img_selector is not None else not is_mask
            if is_mask or is_image:
                self.convert_nii2nii(img_path, is_mask=is_mask)
        new_paths = glob.glob(os.path.join(self.source_path, "**/*.nii.gz"), recursive=True)
        return new_paths

    def convert_nii2nii(self, img_path: str, is_mask: bool = False) -> None:
        """Standardize a single NIfTI volume in place, preserving its geometry.

        Args:
            img_path (str): Path to the source ``.nii.gz`` volume.
            is_mask (bool): If True, the volume is a segmentation mask and is passed through
                untouched (never windowed).
        """
        nii_img = nib.load(img_path)
        # Masks must never be windowed - leave the volume (and its full header) untouched.
        if is_mask:
            return
        # Image volumes are optionally windowed when both window parameters are configured.
        if self.window_center is None or self.window_width is None:
            return
        data = nii_img.get_fdata()  # type: ignore[attr-defined]
        windowed = self._apply_window(data)
        new_header = nii_img.header.copy()  # type: ignore[attr-defined]
        new_header.set_data_dtype(windowed.dtype)  # type: ignore[attr-defined]  # nibabel stub gap
        out = nib.Nifti1Image(windowed, affine=nii_img.affine, header=new_header)  # type: ignore[no-untyped-call,attr-defined]
        nib.save(out, img_path)

    def _apply_window(self, pixel_data: np.ndarray) -> np.ndarray:
        """Clip a volume to the configured window and rescale it to the 0-255 range.

        Mirrors the windowing used by :class:`ConvertNii2Png` so 2D slices and 3D volumes
        share identical intensity normalization.

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
        # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
        min_val = np.min(pixel_data)
        min_val = -1000 if min_val < -1000 else min_val
        pixel_data = pixel_data - min_val
        ratio = np.max(pixel_data) / 255
        if ratio == 0:
            ratio = 1
        pixel_data = np.divide(pixel_data, ratio)
        return pixel_data.astype(np.uint8)
