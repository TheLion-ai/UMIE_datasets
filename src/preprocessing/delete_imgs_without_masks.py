"""Delete images without masks."""
import glob
import os

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class DeleteImgsWithoutMasks(BaseEstimator, TransformerMixin):
    """Delete images without masks."""
    def __init__(self, mask_folder_name: str = "Masks", **kwargs):
        self.mask_folder_name = mask_folder_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: str):
        """Delete images without masks.
        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.dirname(os.path.dirname(X[0]))
        mask_paths = glob.glob(
            f"{os.path.join(root_path, self.mask_folder_name)}/**/*.png", recursive=True
        )
        mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Deleting images without masks...")
        for img_path in tqdm(X):
            img_name = os.path.basename(img_path)
            if img_name not in mask_names:
                self.delete_imgs_without_masks(img_path)

        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(f"{root_path}/*.png", recursive=True)
        return new_paths

    def delete_imgs_without_masks(self, img_path: str):
        """Delete images without masks.
        Args:
            img_path (str): Path to the image.
        """
        os.remove(img_path)
