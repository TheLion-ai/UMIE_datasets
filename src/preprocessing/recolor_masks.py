"""Recolors masks from default color to the color specified in the config."""
import glob
import os

import cv2
import numpy as np
import tqdm
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


class RecolorMasks(BaseEstimator, TransformerMixin):
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
            source_path (str): path to the folder with masks
            target_path (str): path to the folder where recolored masks will be saved
            mask (str): name of the mask, check masks config for the list of available masks and add new ones if needed
            extension (str): extension of the mask files (only images supported)
            source_color (int): color of the mask to be changed
        """
        root_path = os.path.join(
            os.path.dirname(os.path.dirname(X)), self.mask_folder_name
        )
        mask_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        for mask_path in mask_paths:
            self.recolor_masks(mask_path)
        return X

    def recolor_masks(self, mask_path):
        mask = cv2.imread(mask_path)
        # changing pixel values
        for source_color, target_color in self.mask_colors_old2new.items():
            np.place(mask, mask == source_color, target_color)
        cv2.imwrite(mask_path, mask)
