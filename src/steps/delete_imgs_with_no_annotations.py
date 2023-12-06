"""Delete images without masks."""
import glob
import os

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class DeleteImgsWithNoAnnotations(TransformerMixin):
    """Delete images without masks."""

    def __init__(
        self,
        mask_folder_name: str = "Masks",
        **kwargs: dict,
    ):
        """Delete images without masks.

        Args:
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.mask_folder_name = mask_folder_name

    def transform(self, X: str) -> list:
        """Delete images without masks.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.dirname(os.path.dirname(X[0]))
        mask_paths = glob.glob(f"{os.path.join(root_path, self.mask_folder_name)}/**/*.png", recursive=True)
        self.mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Deleting images without masks...")

        for img_path in tqdm(X):
            self.delete_imgs_with_no_annotations(img_path, root_path)

        # Create new list of paths after the deletion
        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)
        return new_paths

    def delete_imgs_with_no_annotations(self, img_path: str, root_path: str) -> None:
        """Delete images without any annotations i.e. labels or masks.

        Args:
            img_path (str): Path to the image.
            root_path (str): Path to the root folder.
        """
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(root_path, self.mask_folder_name, img_name)
        no_label = False
        if "-" not in img_name:
            no_label = True
        if img_name not in self.mask_names:
            if no_label:
                # If there is no mask and no label
                os.remove(img_path)
        # If there is a mask but it is empty
        elif np.unique(cv2.imread(mask_path)).shape[0] == 1:
            if no_label:
                # If there is a blank mask and no label
                os.remove(img_path)
                os.remove(mask_path)
