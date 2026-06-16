"""Optional, opt-in CT intensity windowing applied to image PNGs only (Theme E, Task 14)."""

import glob
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode

# Named CT windows as (center, width) in Hounsfield units. Reused by ApplyWindowing to resolve
# self.preprocessing.window_preset; values are the conventional radiology presets.
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "lung": (-600, 1500),
    "mediastinum": (40, 400),
    "abdomen": (50, 400),
    "bone": (400, 1800),
    "brain": (40, 80),
    "soft_tissue": (50, 400),
}


class ApplyWindowing(BaseStep):
    """Clip image intensities to a CT window and rescale to 0-255 uint8 (images only)."""

    def transform(self, X: list) -> list:
        """Apply the resolved CT window to every image PNG, leaving masks untouched.

        The window is resolved from ``self.preprocessing.window_preset`` (a named preset), or
        from ``self.window_center``/``self.window_width`` (DICOM config) as a fallback. When no
        window can be resolved the step is a no-op and returns ``X`` unchanged. Only runs in 2D
        (PNG) output mode; 3D volumes are windowed during conversion instead.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        window = self._resolve_window()
        if window is None or self.output_mode == OutputMode.VOLUMES_3D:
            return X

        center, width = window
        image_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.png"), recursive=True)
        print("Applying CT windowing to images...")
        for image_path in image_paths:
            self._apply_windowing(image_path, center, width)
        return X

    def _resolve_window(self) -> Optional[tuple[float, float]]:
        """Resolve the active (center, width); preset wins, then DICOM config, else None.

        Returns:
            Optional[tuple[float, float]]: The (center, width) to apply, or None for a no-op.
        """
        preset = self.preprocessing.window_preset
        if preset is not None:
            if preset not in WINDOW_PRESETS:
                raise ValueError(f"Unknown window preset '{preset}'. Choose from {sorted(WINDOW_PRESETS)}.")
            center, width = WINDOW_PRESETS[preset]
            return float(center), float(width)
        if self.window_center is not None and self.window_width is not None:
            return float(self.window_center), float(self.window_width)
        return None

    def _apply_windowing(self, image_path: str, center: float, width: float) -> None:
        """Clip a single image to the window and rescale linearly to 0-255 uint8.

        Reads with ``cv2.IMREAD_UNCHANGED`` so the source bit depth is preserved before mapping.

        Args:
            image_path (str): Path to the image PNG.
            center (float): Window center in source intensity units.
            width (float): Window width in source intensity units.
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        lower = center - width / 2
        upper = center + width / 2
        clipped = np.clip(image.astype(np.float64), lower, upper)
        # Linear map [lower, upper] -> [0, 255]; width>0 by construction for the presets.
        scale = 255.0 / width if width != 0 else 0.0
        rescaled = ((clipped - lower) * scale).round().astype(np.uint8)
        cv2.imwrite(image_path, rescaled)
