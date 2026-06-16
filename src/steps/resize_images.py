"""Optional, opt-in image resizing keeping the paired mask aligned (Theme E, Task 17)."""

import glob
import os

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class ResizeImages(BaseStep):
    """Resize images to a target (h, w) with a chosen strategy; masks use nearest-neighbour."""

    def transform(self, X: list) -> list:
        """Resize every image PNG to ``self.preprocessing.target_size`` and align its mask.

        Images are resampled bilinearly (``cv2.INTER_LINEAR``); the paired mask is resampled
        nearest-neighbour (``cv2.INTER_NEAREST``) with the SAME geometry so it stays pixel-aligned
        and gains no new label values. No-op when ``target_size`` is None. Only runs in 2D mode.

        Args:
            X (list): List of paths to the images.

        Returns:
            list: The unchanged list of image paths.
        """
        target_size = self.preprocessing.target_size
        if target_size is None or self.output_mode == OutputMode.VOLUMES_3D:
            return X

        target = (int(target_size[0]), int(target_size[1]))
        strategy = self.preprocessing.resize_strategy
        image_paths = glob.glob(os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.png"), recursive=True)
        print("Resizing images...")
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            cv2.imwrite(image_path, self._resize(image, target, strategy, cv2.INTER_LINEAR))

            mask_path = image_path.replace(
                os.sep + self.image_folder_name + os.sep, os.sep + self.mask_folder_name + os.sep
            )
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    cv2.imwrite(mask_path, self._resize(mask, target, strategy, cv2.INTER_NEAREST))
        return X

    def _resize(self, image: np.ndarray, target: tuple, strategy: str, interpolation: int) -> np.ndarray:
        """Resize one array to ``target`` (height, width) using the requested strategy.

        Args:
            image (np.ndarray): Source image or mask array.
            target (tuple): Desired (height, width).
            strategy (str): One of ``pad``, ``crop``, ``letterbox``, ``stretch``.
            interpolation (int): cv2 interpolation flag (bilinear for images, nearest for masks).

        Returns:
            np.ndarray: The resized array with shape ``target`` (plus any channel axis).
        """
        target_h, target_w = target
        if strategy == "stretch":
            # Ignore aspect ratio: scale straight to the target.
            return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        if strategy == "letterbox":
            # Scale to fit inside the target preserving aspect ratio, then zero-pad.
            scale = min(target_h / image.shape[0], target_w / image.shape[1])
            resized = self._scale(image, scale, interpolation)
            return self._fit_canvas(resized, target_h, target_w)
        if strategy == "pad":
            # No scaling: center on a zero canvas, center-cropping any axis that overflows.
            return self._fit_canvas(image, target_h, target_w)
        if strategy == "crop":
            # Scale to cover the target preserving aspect ratio, then center-crop the overflow.
            scale = max(target_h / image.shape[0], target_w / image.shape[1])
            resized = self._scale(image, scale, interpolation)
            return self._fit_canvas(resized, target_h, target_w)
        raise ValueError(f"Unknown resize strategy '{strategy}'. Choose from pad, crop, letterbox, stretch.")

    @staticmethod
    def _scale(image: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
        """Uniformly scale an image by ``scale`` (at least 1px per axis), preserving aspect ratio."""
        new_h = max(1, int(round(image.shape[0] * scale)))
        new_w = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    @staticmethod
    def _fit_canvas(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Center the image on a ``target_h`` x ``target_w`` zero canvas, cropping any overflow.

        Always returns an array of exactly the target spatial dimensions (plus any channel axis),
        so every strategy yields the requested size regardless of the source size.
        """
        source_h, source_w = image.shape[:2]
        crop_h = min(source_h, target_h)
        crop_w = min(source_w, target_w)
        src_top = (source_h - crop_h) // 2
        src_left = (source_w - crop_w) // 2
        cropped = image[src_top : src_top + crop_h, src_left : src_left + crop_w]

        canvas_shape = (target_h, target_w) + image.shape[2:]
        canvas = np.zeros(canvas_shape, dtype=image.dtype)
        dst_top = (target_h - crop_h) // 2
        dst_left = (target_w - crop_w) // 2
        canvas[dst_top : dst_top + crop_h, dst_left : dst_left + crop_w] = cropped
        return canvas
