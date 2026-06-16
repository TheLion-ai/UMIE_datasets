"""Optional, opt-in auto-cropping of uniform black borders, keeping masks aligned (Theme E, Task 19)."""

import glob
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class AutocropBorders(BaseStep):
    """Crop uniform dark borders from images, applying the identical crop box to paired masks."""

    def transform(self, X: list) -> list:
        """Crop each image to its content bounding box and crop the paired mask identically.

        Background is any pixel ``<= self.preprocessing.autocrop_tolerance``. The content bounding
        box is computed from the image and the SAME box is applied to the paired mask so they stay
        aligned. No-op when ``autocrop_enabled`` is False. Only runs in 2D (PNG) output mode.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        if not self.preprocessing.autocrop_enabled or self.output_mode == OutputMode.VOLUMES_3D:
            return X

        tolerance = self.preprocessing.autocrop_tolerance
        image_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.png"), recursive=True)
        print("Auto-cropping image borders...")
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            bbox = self._content_bbox(image, tolerance)
            if bbox is None:
                continue  # fully-background image: leave it untouched
            top, bottom, left, right = bbox
            cv2.imwrite(image_path, image[top:bottom, left:right])

            mask_path = image_path.replace(
                os.sep + self.image_folder_name + os.sep, os.sep + self.mask_folder_name + os.sep
            )
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    cv2.imwrite(mask_path, mask[top:bottom, left:right])
        return X

    @staticmethod
    def _content_bbox(image: np.ndarray, tolerance: int) -> Optional[tuple[int, int, int, int]]:
        """Compute the half-open (top, bottom, left, right) bounding box of non-background pixels.

        Args:
            image (np.ndarray): Image array (single- or multi-channel).
            tolerance (int): Pixels with value <= this are treated as background.

        Returns:
            Optional[tuple[int, int, int, int]]: The bounding box, or None if all-background.
        """
        # Collapse channels so any channel above tolerance marks the pixel as content.
        intensity = image if image.ndim == 2 else image.max(axis=2)
        content = intensity > tolerance
        rows = np.any(content, axis=1)
        cols = np.any(content, axis=0)
        if not rows.any() or not cols.any():
            return None
        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]
        top, bottom = int(row_idx[0]), int(row_idx[-1]) + 1
        left, right = int(col_idx[0]), int(col_idx[-1]) + 1
        return top, bottom, left, right
