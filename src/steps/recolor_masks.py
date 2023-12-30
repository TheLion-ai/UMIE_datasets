"""Recolors masks from default color to the color specified in the config."""
import glob
import os
from typing import Callable

import cv2
import numpy as np
from sklearn.base import TransformerMixin


class RecolorMasks(TransformerMixin):
    """Recolors masks from default color to the color specified in the config."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        mask_colors_source2target: dict,
        mask_folder_name: str = "Masks",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        **kwargs: dict,
    ):
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_colors_source2target (dict): Dictionary with old and new colors.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.mask_colors_source2target = mask_colors_source2target
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
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
        for source_color, target_color in self.mask_colors_source2target.items():
            np.place(mask, mask == source_color, target_color)
        cv2.imwrite(mask_path, mask)
