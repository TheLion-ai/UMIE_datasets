"""Optional, opt-in CLAHE contrast enhancement applied to image PNGs only (Theme E, Task 15)."""

import glob
import os

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class ApplyClahe(BaseStep):
    """Apply Contrast Limited Adaptive Histogram Equalization to images (opt-in, images only)."""

    def transform(self, X: list) -> list:
        """Apply CLAHE to every image PNG when enabled; otherwise a no-op.

        When ``self.preprocessing.clahe_enabled`` is False the step writes nothing and returns
        ``X`` unchanged, so output files stay byte-identical. Masks are never touched. Only runs
        in 2D (PNG) output mode.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        if not self.preprocessing.clahe_enabled or self.output_mode == OutputMode.VOLUMES_3D:
            return X

        clahe = cv2.createCLAHE(
            clipLimit=self.preprocessing.clahe_clip_limit,
            tileGridSize=tuple(self.preprocessing.clahe_tile_grid_size),
        )
        image_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.png"), recursive=True)
        print("Applying CLAHE to images...")
        for image_path in image_paths:
            self._apply_clahe(image_path, clahe)
        return X

    def _apply_clahe(self, image_path: str, clahe: "cv2.CLAHE") -> None:
        """Apply CLAHE to a single image, keeping the output a valid uint8 PNG.

        Multi-channel images are equalized per channel so the result stays a readable PNG.

        Args:
            image_path (str): Path to the image PNG.
            clahe (cv2.CLAHE): The configured CLAHE operator.
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        # CLAHE operates on 8-bit (or 16-bit) single-channel data; downcast 16-bit to 8-bit so the
        # output is a standard uint8 PNG, then equalize each channel independently.
        if image.dtype != np.uint8:
            normalized = np.empty_like(image, dtype=np.float64)
            image = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if image.ndim == 2:
            result = clahe.apply(image)
        else:
            channels = [clahe.apply(image[:, :, c]) for c in range(image.shape[2])]
            result = cv2.merge(channels)
        cv2.imwrite(image_path, result)
