"""Recolors masks from shades of gray to 2 colors."""
import glob
import os
from typing import Callable

import cv2
from tqdm import tqdm

from base.step import BaseStep


class MasksToBinaryColors(BaseStep):
    """Recolors masks from shades of gray to 2 colors."""

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        mask_paths = glob.glob(os.path.join(root_path, f"**/{self.mask_folder_name}/*.png"), recursive=True)

        for mask_path in tqdm(mask_paths):
            if os.path.exists(mask_path):
                self.scale_to_2_colors(mask_path)
        return X

    def scale_to_2_colors(self, mask_path: str) -> None:
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_path (str): Path to the mask.
        """
        mask = cv2.imread(mask_path)
        (thresh, blackAndWhiteMask) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # changing pixel values
        cv2.imwrite(mask_path, blackAndWhiteMask)
