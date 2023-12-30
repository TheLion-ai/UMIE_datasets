"""Recolors masks from shades of gray to 2 colors."""
import glob
import os
from typing import Callable

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class MasksToBinaryColors(TransformerMixin):
    """Recolors masks from shades of gray to 2 colors."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        mask_folder_name: str,
        image_folder_name: str,
        img_prefix: str,
        segmentation_prefix: str,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        **kwargs: dict,
    ):
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_colors_old2new (dict): Dictionary with old and new colors.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            segmentation_prefix (str, optional): String to select masks. Defaults to "segmentations".
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.img_id_extractor = img_id_extractor
        self.mask_folder_name = mask_folder_name
        self.image_folder_name = image_folder_name
        self.img_prefix = img_prefix
        self.segmentation_prefix = segmentation_prefix

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
