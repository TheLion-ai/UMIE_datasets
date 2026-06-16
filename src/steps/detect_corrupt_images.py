"""Detect corrupt, truncated, blank or undersized output images and write a report."""

import glob
import json
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class DetectCorruptImages(BaseStep):
    """Detect corrupt, truncated, blank or undersized output images and write a report."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Verify each output image is readable, large enough and not blank; write a report.

        Each output image (PNG, or ``.nii.gz`` in 3D mode) is categorised as ``unreadable``,
        ``truncated``, ``blank`` (pixel std below ``quality.blank_std_threshold``) or
        ``too_small`` (below ``quality.expected_min_size``). The categories and a flat
        quarantine list are written to a JSON report. ``X`` is returned unchanged; nothing is
        ever deleted.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: The unchanged list of image paths.
        """
        print("Detecting corrupt images...")
        report: dict = {
            "unreadable": [],
            "truncated": [],
            "blank": [],
            "too_small": [],
            "quarantine": [],
        }
        for path in self._output_image_paths():
            self._inspect(path, report)
        self._write_report(report)
        print(f"Corrupt-image detection complete: {len(report['quarantine'])} image(s) quarantined.")
        return X

    def _output_image_paths(self) -> list:
        """Glob the dataset's output images (PNG in 2D mode, ``.nii.gz`` in 3D mode)."""
        extension = "nii.gz" if self.output_mode == OutputMode.VOLUMES_3D else "png"
        pattern = os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.{extension}")
        return sorted(glob.glob(pattern, recursive=True))

    def _read_array(self, path: str) -> Optional[np.ndarray]:
        """Read an output image or volume into a numpy array, or None if unreadable.

        Args:
            path (str): Path to the output image or volume.
        Returns:
            Optional[np.ndarray]: The pixel array, or None when it cannot be read.
        """
        if path.endswith(".nii.gz"):
            try:
                import nibabel as nib  # type: ignore[import-untyped]

                return np.asarray(nib.load(path).get_fdata())  # type: ignore[attr-defined]
            except Exception:
                return None
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            return None
        return image

    def _inspect(self, path: str, report: dict) -> None:
        """Classify a single output image and append it to the relevant report buckets.

        Args:
            path (str): Path to the output image.
            report (dict): Mutable report accumulator.
        """
        umie_path = self.get_path_without_target_path(path)
        array = self._read_array(path)

        # An unreadable file is either genuinely corrupt or truncated; distinguish by whether
        # the file has any bytes at all so empty/clipped outputs are reported as "truncated".
        if array is None or getattr(array, "size", 0) == 0:
            category = "truncated" if os.path.getsize(path) == 0 else "unreadable"
            report[category].append(umie_path)
            report["quarantine"].append(umie_path)
            return

        if array.ndim >= 2:
            height, width = array.shape[0], array.shape[1]
            min_size = self.quality.expected_min_size
            if min_size is not None and (height < min_size[0] or width < min_size[1]):
                report["too_small"].append(umie_path)
                report["quarantine"].append(umie_path)

        if float(np.std(array)) < self.quality.blank_std_threshold:
            report["blank"].append(umie_path)
            if umie_path not in report["quarantine"]:
                report["quarantine"].append(umie_path)

    def _write_report(self, report: dict) -> None:
        """Write the corrupt-image report to JSON under the reports dir.

        Args:
            report (dict): The accumulated report.
        """
        report_path = os.path.join(self.reports_dir(), "corrupt_images_report.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)
