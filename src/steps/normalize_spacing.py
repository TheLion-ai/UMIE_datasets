"""Optional, opt-in pixel-spacing normalization for NIfTI volumes (Theme E, Task 16)."""

import glob
import os
from typing import Optional

import jsonlines
import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
from scipy.ndimage import zoom  # type: ignore[import-untyped]

from base.step import BaseStep
from constants import OutputMode


class NormalizeSpacing(BaseStep):
    """Resample NIfTI volumes to a target voxel spacing (images bilinear, masks nearest)."""

    def transform(self, X: list) -> list:
        """Resample image and mask volumes to ``self.preprocessing.target_spacing_mm``.

        Primary path is 3D NIfTI volumes: the current zooms are read from the header, the data is
        resampled with ``scipy.ndimage.zoom`` (order=1 for images, order=0 for masks so no new
        label values appear), the affine scale is updated, and the new spacing is recorded in the
        JSONL as an additive ``pixel_spacing_mm`` field. 2D PNGs carry no spacing source and are
        skipped (no-op). No-op when ``target_spacing_mm`` is None.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        target = self.preprocessing.target_spacing_mm
        if target is None or self.output_mode != OutputMode.VOLUMES_3D:
            # No spacing source for 2D PNG output; documented skip.
            return X

        target_spacing = tuple(float(v) for v in target)
        new_spacings: dict[str, list[float]] = {}

        image_paths = glob.glob(
            os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.nii.gz"), recursive=True
        )
        mask_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.mask_folder_name}/*.nii.gz"), recursive=True)
        print("Normalizing voxel spacing...")
        for image_path in image_paths:
            new_spacings[image_path] = self._resample(image_path, target_spacing, is_mask=False)
        for mask_path in mask_paths:
            self._resample(mask_path, target_spacing, is_mask=True)

        self._record_spacing(new_spacings)
        return X

    def _resample(self, volume_path: str, target_spacing: tuple, is_mask: bool) -> list[float]:
        """Resample a single NIfTI volume to ``target_spacing`` and overwrite it in place.

        Args:
            volume_path (str): Path to the ``.nii.gz`` volume.
            target_spacing (tuple): Desired voxel spacing in mm (one value per spatial axis).
            is_mask (bool): If True use nearest-neighbour (order=0) so no new labels appear.

        Returns:
            list[float]: The resulting voxel spacing actually written to the header.
        """
        nii = nib.load(volume_path)
        data = np.asarray(nii.get_fdata())  # type: ignore[attr-defined]
        current_spacing = tuple(float(z) for z in nii.header.get_zooms()[: data.ndim])  # type: ignore[attr-defined]
        factors = tuple(current_spacing[i] / target_spacing[i] for i in range(data.ndim))

        order = 0 if is_mask else 1
        resampled = zoom(data, factors, order=order)
        # Masks keep their original integer dtype so order=0 introduces no new label values; image
        # data may stay floating-point.
        out_dtype = nii.get_data_dtype() if is_mask else resampled.dtype  # type: ignore[attr-defined]
        resampled = resampled.astype(out_dtype)

        affine = np.array(nii.affine, dtype=np.float64)  # type: ignore[attr-defined]
        new_affine = affine.copy()
        # Rescale each spatial column of the affine so the new spacing is encoded, preserving
        # the rotation/orientation direction of each axis.
        for axis in range(min(3, data.ndim)):
            column = affine[:3, axis]
            norm = float(np.linalg.norm(column))
            if norm != 0:
                new_affine[:3, axis] = column / norm * target_spacing[axis]

        new_header = nii.header.copy()  # type: ignore[attr-defined]
        new_header.set_data_dtype(out_dtype)  # type: ignore[attr-defined]
        # Clear any intensity scaling so the resampled values are written verbatim (otherwise a
        # stale scl_slope/scl_inter from the source header would re-quantize the data on save).
        new_header.set_slope_inter(None, None)  # type: ignore[attr-defined]
        new_header.set_zooms(tuple(target_spacing[: data.ndim]))  # type: ignore[attr-defined]
        out = nib.Nifti1Image(resampled, affine=new_affine, header=new_header)  # type: ignore[no-untyped-call]
        nib.save(out, volume_path)
        return [float(z) for z in out.header.get_zooms()[: data.ndim]]  # type: ignore[no-untyped-call]

    def _record_spacing(self, new_spacings: dict[str, list[float]]) -> None:
        """Add an additive ``pixel_spacing_mm`` field to matching JSONL records.

        Existing fields are never overwritten; records whose volume was not resampled are left
        unchanged. Does nothing when the JSONL file does not yet exist.

        Args:
            new_spacings (dict[str, list[float]]): Map of absolute image path -> new spacing.
        """
        if not os.path.exists(self.json_path) or not new_spacings:
            return
        by_relpath = {self.get_path_without_target_path(path): spacing for path, spacing in new_spacings.items()}
        with jsonlines.open(self.json_path, mode="r") as reader:
            records = list(reader)
        for record in records:
            spacing = self._lookup_spacing(record, by_relpath)
            if spacing is not None and "pixel_spacing_mm" not in record:
                record["pixel_spacing_mm"] = spacing
        with jsonlines.open(self.json_path, mode="w") as writer:
            for record in records:
                writer.write(record)

    @staticmethod
    def _lookup_spacing(record: dict, by_relpath: dict[str, list[float]]) -> Optional[list[float]]:
        """Find the new spacing for a JSONL record by its ``umie_path`` (basename fallback).

        Args:
            record (dict): A single JSONL record.
            by_relpath (dict[str, list[float]]): Map of relative image path -> new spacing.

        Returns:
            Optional[list[float]]: The matching spacing, or None when no volume matches.
        """
        umie_path = record.get("umie_path")
        if umie_path in by_relpath:
            return by_relpath[umie_path]
        for relpath, spacing in by_relpath.items():
            if umie_path is not None and os.path.basename(relpath) == os.path.basename(str(umie_path)):
                return spacing
        return None
