"""Converts mhd and raw files to png images."""

import glob
import os
from typing import Callable

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from base.extractors.img_id import BaseImgIdExtractor
from base.step import BaseStep


class ConvertMhd2Png(BaseStep):
    """Converts mhd and raw files to png images."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert mhd and raw files to png images.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Converting mhd to png...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            if img_path.endswith(".mhd"):
                if self.segmentation_prefix in img_path or self.img_selector(img_path):
                    self.convert_mhd2png(img_path)

        mask_paths = []
        for root, _, filenames in os.walk(self.masks_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    path = os.path.join(root, filename)
                    # Verify if file is a mask
                    if self.mask_selector(path):
                        mask_paths.append(path)
        if mask_paths:
            for mask_path in mask_paths:
                if mask_path.endswith(".mhd"):
                    self.convert_mhd2png(mask_path)
        else:
            print("Masks not found.")

        new_paths = glob.glob(os.path.join(self.source_path, f"**/{self.img_prefix}*.png"), recursive=True)
        return new_paths

    def convert_mhd2png(self, img_path: str) -> None:
        """Convert mhd and raw files to png images.

        Args:
            img_path (str): Path to the image.
        """
        try:
            image = sitk.ReadImage(img_path)
            array = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]

            for idx, slice_ in enumerate(array):
                # Normalize to 0–255 for saving
                slice_norm = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX)
                slice_uint8 = slice_norm.astype(np.uint8)
                root_path = os.path.dirname(img_path)
                name = os.path.basename(img_path).split(".")[0] + f"_{str(idx).zfill(self.zfill)}.png"
                new_path = os.path.join(root_path, name)

                cv2.imwrite(new_path, slice_uint8)
        except Exception as e:
            print(f"Error {e} occured while converting {img_path}")
            if self.on_error_remove:
                os.remove(img_path)
