"""Delete images with empty masks."""

import glob
import os

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class DeleteImgsWithEmptyMasks(TransformerMixin):
    """Delete images with empty masks."""

    def __init__(
        self,
        mask_folder_name: str,
        **kwargs: dict,
    ):
        """Delete images with empty masks.

        Args:
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.mask_folder_name = mask_folder_name

    def transform(self, X: str) -> list:
        """Delete images with empty masks.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.dirname(os.path.dirname(X[0]))
        print("Deleting images with empty masks...")

        for img_path in tqdm(X):
            self.delete_imgs_with_empty_masks(img_path, root_path)

        # Create new list of paths after the deletion
        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)
        return new_paths

    def delete_imgs_with_empty_masks(self, img_path: str, root_path: str) -> None:
        """Delete images without any annotations i.e. labels or masks.

        Args:
            img_path (str): Path to the image.
            root_path (str): Path to the root folder.
        """
        img_name = os.path.basename(img_path)
        if self.mask_folder_name:
            mask_path = os.path.join(root_path, self.mask_folder_name, img_name)
        if self.mask_folder_name:
            mask = cv2.imread(mask_path)
            if not np.any(mask):
                os.remove(mask_path)
                os.remove(img_path)
