"""Recolors masks from default color to the color specified in the config."""
import glob
import os

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class RecolorMasks(TransformerMixin):
    """Recolors masks from default color to the color specified in the config."""

    def __init__(
        self,
        mask_colors_source2target: dict,
        mask_folder_name: str = "Masks",
        **kwargs: dict,
    ):
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_colors_source2target (dict): Dictionary with old and new colors.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.mask_colors_source2target = mask_colors_source2target
        self.mask_folder_name = mask_folder_name

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        root_path = os.path.join(os.path.dirname(os.path.dirname(X[0])), self.mask_folder_name)
        print("Recoloring masks...")
        mask_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        for mask_path in tqdm(mask_paths):
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
