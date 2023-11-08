"""Recolors masks from default color to the color specified in the config."""
import glob
import os

import cv2
import numpy as np
import tqdm
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


class RecolorMasks(BaseEstimator, TransformerMixin):
    """Recolors masks from default color to the color specified in the config."""
    def __init__(
        self, mask_colors_old2new: dict, mask_folder_name: str = "Masks", **kwargs
    ):
        self.mask_colors_old2new = mask_colors_old2new
        self.mask_folder_name = mask_folder_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Recolors masks from default color to the color specified in the config.
        UMIE datasets consists of masks from several opensource datasets. Each type of masks has unique color encoding.
        To find if the mask has encoding, check the config file. If the mask is not in the config, add it.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.join(
            os.path.dirname(os.path.dirname(X)), self.mask_folder_name
        )
        mask_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        for mask_path in mask_paths:
            self.recolor_masks(mask_path)
        return X

    def recolor_masks(self, mask_path):
        """Recolors masks from default color to the color specified in the config.
        Args:
            mask_path (str): Path to the mask.
        """
        mask = cv2.imread(mask_path)
        # changing pixel values
        for source_color, target_color in self.mask_colors_old2new.items():
            np.place(mask, mask == source_color, target_color)
        cv2.imwrite(mask_path, mask)
