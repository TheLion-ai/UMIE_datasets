"""Check output mask quality: dimensions, colour vocabulary and emptiness."""

import glob
import json
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from config.masks import all_masks
from constants import OutputMode


class CheckMaskQuality(BaseStep):
    """Check output mask quality: dimensions, colour vocabulary and emptiness."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Validate each output mask's dimensions, colours and emptiness; write a report.

        For every output mask the step checks that its dimensions match the paired image, that
        every unique pixel value is in the allowed vocabulary (``{0}`` plus the configured
        ``target_color`` values, each of which must be a valid colour in ``config/masks.py``)
        and flags all-zero (empty) masks. Failures are listed in a JSON report; nothing is
        dropped. ``X`` is returned unchanged.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: The unchanged list of image paths.
        """
        print("Checking mask quality...")
        allowed_colors = self._allowed_colors()
        report: dict = {
            "allowed_colors": sorted(allowed_colors),
            "dim_mismatch": [],
            "out_of_vocab_color": [],
            "empty": [],
        }
        for mask_path in self._output_mask_paths():
            self._inspect(mask_path, allowed_colors, report)
        self._write_report(report)
        failures = len(report["dim_mismatch"]) + len(report["out_of_vocab_color"]) + len(report["empty"])
        print(f"Mask-quality check complete: {failures} issue(s) found.")
        return X

    def _output_mask_paths(self) -> list:
        """Glob the dataset's output masks (PNG in 2D mode, ``.nii.gz`` in 3D mode)."""
        extension = "nii.gz" if self.output_mode == OutputMode.VOLUMES_3D else "png"
        pattern = os.path.join(self.dataset_root, f"**/{self.mask_folder_name}/*.{extension}")
        return sorted(glob.glob(pattern, recursive=True))

    def _allowed_colors(self) -> set:
        """Build the allowed colour set: 0 plus configured target colours valid in config/masks.

        Returns:
            set: The allowed pixel values for any output mask.
        """
        valid_colors = {mask.color for mask in all_masks}
        allowed = {0}
        for mask_color in self.masks.values():
            target = mask_color["target_color"]
            if target in valid_colors:
                allowed.add(target)
        return allowed

    def _read_array(self, path: str) -> Optional[np.ndarray]:
        """Read a mask image or volume into a numpy array, or None if unreadable.

        Args:
            path (str): Path to the output mask.
        Returns:
            Optional[np.ndarray]: The pixel array, or None when it cannot be read.
        """
        if path.endswith(".nii.gz"):
            try:
                import nibabel as nib  # type: ignore[import-untyped]

                return np.asarray(nib.load(path).get_fdata())  # type: ignore[attr-defined]
            except Exception:
                return None
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _paired_image_array(self, mask_path: str) -> Optional[np.ndarray]:
        """Read the image paired with a mask (same basename under the image folder).

        Args:
            mask_path (str): Path to the output mask.
        Returns:
            Optional[np.ndarray]: The paired image array, or None when missing/unreadable.
        """
        sep = os.sep + self.mask_folder_name + os.sep
        replacement = os.sep + self.image_folder_name + os.sep
        image_path = mask_path.replace(sep, replacement)
        if not os.path.exists(image_path):
            return None
        return self._read_array(image_path)

    def _inspect(self, mask_path: str, allowed_colors: set, report: dict) -> None:
        """Classify a single output mask and record any failures in the report.

        Args:
            mask_path (str): Path to the output mask.
            allowed_colors (set): The allowed pixel values.
            report (dict): Mutable report accumulator.
        """
        umie_path = self.get_path_without_target_path(mask_path)
        mask = self._read_array(mask_path)
        if mask is None:
            report["out_of_vocab_color"].append({"mask": umie_path, "reason": "unreadable"})
            return

        image = self._paired_image_array(mask_path)
        if image is not None and image.shape[:2] != mask.shape[:2]:
            report["dim_mismatch"].append(
                {"mask": umie_path, "mask_shape": list(mask.shape[:2]), "image_shape": list(image.shape[:2])}
            )

        unique_values = {int(value) for value in np.unique(mask)}
        invalid = sorted(unique_values - allowed_colors)
        if invalid:
            report["out_of_vocab_color"].append({"mask": umie_path, "invalid_colors": invalid})

        if unique_values == {0}:
            report["empty"].append(umie_path)

    def _write_report(self, report: dict) -> None:
        """Write the mask-quality report to JSON under the reports dir.

        Args:
            report (dict): The accumulated report.
        """
        report_path = os.path.join(self.reports_dir(), "mask_quality_report.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)
