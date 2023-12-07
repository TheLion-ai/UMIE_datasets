"""Add labels to the images and masks based on the labels.json file. The step requires the pipeline to specify the function for mapping the images with annotations."""
import glob
import json
import os
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddLabels(TransformerMixin):
    """Add labels to the images and masks based on the function for mapping the images with annotations specified by the pipeline."""

    def __init__(
        self,
        mask_folder_name: str = "Masks",
        get_label: Callable = lambda x: [],
        **kwargs: dict,
    ):
        """Add labels to the images and masks based on the labels.json file.

        Args:
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            get_label (Callable, optional): Function to get the label. Defaults to lambda x: [].
        """
        self.mask_folder_name = mask_folder_name
        self.get_label = get_label

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Add labels to the images and masks.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Adding labels...")
        for img_path in tqdm(X):
            self.add_labels(img_path)
        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)
        return new_paths

    def add_labels(self, img_path: str) -> None:
        """Add labels to the image and mask based on the get_label function specified by the pipeline.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        img_root_path = os.path.dirname(img_path)
        img_id = os.path.basename(img_path).split(".")[0]
        label_prefix = "-"  # each label in the target file name is prefixed with this character
        labels = self.get_label(img_path)
        if labels:
            # Add labels to the image path
            labels_str = "".join([label_prefix + label for label in labels])
            new_name = f"{img_id}_{labels_str}.png"
            os.rename(img_path, os.path.join(img_root_path, new_name))
            # Add labels to the mask path
            root_path = os.path.dirname(os.path.dirname(img_path))
            if self.mask_folder_name:
                mask_path = os.path.join(root_path, self.mask_folder_name, f"{img_id}.png")
                if os.path.exists(mask_path):
                    os.rename(mask_path, os.path.join(root_path, self.mask_folder_name, new_name))
