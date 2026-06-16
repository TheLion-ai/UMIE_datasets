"""Optional, opt-in bit-depth standardization for image PNGs only (Theme E, Task 18)."""

import glob
import os

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class StandardizeBitDepth(BaseStep):
    """Convert images to a target bit depth (8 or 16) without clipping or wraparound."""

    def transform(self, X: list) -> list:
        """Standardize every image PNG to ``self.preprocessing.target_bit_depth``.

        16->8 divides by 256 (drops the low byte, no clip/overflow); 8->16 multiplies by 256.
        Images already at the target depth are skipped. No-op when ``target_bit_depth`` is None.
        Masks are never touched. Only runs in 2D (PNG) output mode.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        target = self.preprocessing.target_bit_depth
        if target is None or self.output_mode == OutputMode.VOLUMES_3D:
            return X
        if target not in (8, 16):
            raise ValueError(f"target_bit_depth must be 8 or 16, got {target}.")

        image_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.png"), recursive=True)
        print(f"Standardizing image bit depth to {target}-bit...")
        for image_path in image_paths:
            self._standardize(image_path, target)
        return X

    def _standardize(self, image_path: str, target: int) -> None:
        """Convert a single image to the target bit depth, logging any conversion performed.

        Args:
            image_path (str): Path to the image PNG.
            target (int): Target bit depth (8 or 16).
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        current = 16 if image.dtype == np.uint16 else 8
        if current == target:
            return
        converted: np.ndarray
        if target == 8:
            # 16->8: integer divide by 256 keeps the high byte; no clipping/overflow possible.
            converted = (image.astype(np.uint16) // 256).astype(np.uint8)
        else:
            # 8->16: multiply by 256 to spread values across the wider range.
            converted = (image.astype(np.uint16) * 256).astype(np.uint16)
        print(f"Converting {os.path.basename(image_path)} from {current}-bit to {target}-bit")
        cv2.imwrite(image_path, converted)
