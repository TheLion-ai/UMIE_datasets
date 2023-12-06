"""Create blank masks for images that don't have masks."""
import glob
import os

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class CreateBlankMasks(TransformerMixin):
    """Create blank masks for images that don't have masks."""

    def __init__(self, mask_folder_name: str = "Masks", **kwargs: dict):
        """Create blank masks for images that don't have masks.

        Args:
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.mask_folder_name = mask_folder_name

    def transform(
        self,
        X: list,
        target_path: str = "",
    ) -> list:
        """Create blank masks for images that don't have masks.

        Args:
            X (list): List of paths to the images.
            target_path (str): Path to the target folder.
        Returns:
            X (list): List of paths to the images.
        """
        mask_paths = glob.glob(
            f"{os.path.join(target_path, self.mask_folder_name)}/**/*.png",
            recursive=True,
        )
        mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Creating blank masks...")
        for img_path in tqdm(X):
            img_name = os.path.basename(img_path)
            if img_name not in mask_names:
                self.create_blank_masks(img_path)
        return X

    def create_blank_masks(self, img_path: str) -> None:
        """Create blank masks for images that don't have masks.

        Args:
            img_path (str): Path to the image.
        """
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        new_path = os.path.dirname(os.path.dirname(img_path))
        new_path = os.path.join(new_path, self.mask_folder_name, img_name)
        mask = np.zeros(img.shape, np.uint8)  # Create a black mask
        cv2.imwrite(new_path, mask)
