"""Converts nii files to png images with appropriate color encoding."""

import glob
import os
from typing import Callable

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from base.extractors.img_id import BaseImgIdExtractor
from base.step import BaseStep


class ConvertNii2Png(BaseStep):
    """Converts nii files to png images with appropriate color encoding."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert nii files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Converting nii to png...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            if img_path.endswith(".nii.gz"):
                if self.segmentation_prefix in img_path or self.img_prefix in img_path:
                    self.convert_nii2png(img_path)
        new_paths = glob.glob(os.path.join(self.source_path, f"**/{self.img_prefix}*.png"), recursive=True)
        return new_paths

    def convert_nii2png(self, img_path: str) -> None:
        """Convert nii files to png images with appropriate color encoding.

        Args:
            img_path (str): Path to the image.
        """
        nii_img = nib.load(img_path)
        nii_data = nii_img.get_fdata()
        slices = nii_data.shape[0]
        for idx in range(slices):
            root_path = os.path.dirname(img_path)
            name = os.path.basename(img_path).split(".")[0] + f"_{str(idx).zfill(self.zfill)}.png"
            new_path = os.path.join(root_path, name)
            img = np.array(nii_data[idx, :, :])
            if self.segmentation_prefix not in new_path:
                img = self._apply_window(img)

            cv2.imwrite(new_path, img)

    def _apply_window(self, pixel_data: np.ndarray) -> np.ndarray:
        """Apply window to the image.

        Args:
            pixel_data (np.ndarray): Image data.
        Returns:
            np.ndarray: Image data with applied window.
        """
        # apply window
        pixel_data = np.clip(
            pixel_data,
            self.window_center - self.window_width / 2,
            self.window_center + self.window_width / 2,
        )
        # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
        min = np.min(pixel_data)
        min = -1000 if min < -1000 else min
        pixel_data = pixel_data - min
        ratio = np.max(pixel_data) / 255
        pixel_data = np.divide(pixel_data, ratio).astype(int)
        return pixel_data
