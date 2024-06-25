"""Recolors masks from default color to the color specified in the config."""

import glob
import os

import cv2
import numpy as np

from base.step import BaseStep


class RecolorMasks(BaseStep):
    """Recolors masks from default color to the color specified in the config."""

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        # Robust to multiple modalities and lack of masks for some images
        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        mask_paths = glob.glob(os.path.join(root_path, f"**/{self.mask_folder_name}/*.png"), recursive=True)
        print("Recoloring masks...")
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                self.recolor_masks(mask_path)
        return X

    def recolor_masks(self, mask_path: str) -> None:
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_path (str): Path to the mask.
        """
        mask = cv2.imread(mask_path)
        # changing pixel values
        for mask_color in self.masks.values():
            np.place(mask, mask == mask_color["source_color"], mask_color["target_color"])
        cv2.imwrite(mask_path, mask)
