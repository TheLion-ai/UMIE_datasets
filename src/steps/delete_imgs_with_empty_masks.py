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
        image_folder_name: str,
        **kwargs: dict,
    ):
        """Delete images with empty masks.

        Args:
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.mask_folder_name = mask_folder_name
        self.image_folder_name = image_folder_name

    def transform(self, X: str) -> list:
        """Delete images with empty masks.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(X[0])))
        print("Deleting images with empty masks...")

        for img_path in tqdm(X):
            self.delete_imgs_with_empty_masks(img_path)

        # Create new list of paths after the deletion
        new_paths = glob.glob(os.path.join(root_path, f"*{self.image_folder_name}*/**png"), recursive=True)
        print(new_paths)
        return new_paths

    def delete_imgs_with_empty_masks(self, img_path: str) -> None:
        """Delete images without any annotations i.e. labels or masks.

        Args:
            img_path (str): Path to the image.
        """
        if self.mask_folder_name:
            mask_path = img_path.replace(self.image_folder_name, self.mask_folder_name)
            mask = cv2.imread(mask_path)
            if not np.any(mask):
                os.remove(mask_path)
                os.remove(img_path)
