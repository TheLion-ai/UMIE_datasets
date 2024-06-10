"""Combine and recolor multiple masks into one."""

import glob
import os
from collections import defaultdict

import cv2
import numpy as np
import PIL.Image
from sklearn.base import TransformerMixin
from tqdm import tqdm


class JoinMasks(TransformerMixin):
    """Join masks saved in multiple folders, which share the same name."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        masks_path: str,
        dataset_name: str,
        dataset_uid: str,
        dataset_masks: dict,
        mask_encodings: dict,
        mask_selector: str,
        multiple_masks_selector: dict,
        img_prefix: str,
        segmentation_prefix: str,
        mask_folder_name: str = "Masks",
        **kwargs: dict,
    ):
        """Combine and recolor multiple masks into one.

        Args:
            mask_colors_source2target (dict): Dictionary with old and new colors.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.source_path = source_path
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.dataset_masks = dataset_masks
        self.mask_encodings = mask_encodings
        self.mask_selector = mask_selector
        self.multiple_masks_selector = multiple_masks_selector
        self.mask_folder_name = mask_folder_name
        self.img_prefix = img_prefix
        self.segmentation_prefix = segmentation_prefix

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        mask_name_to_image_paths = defaultdict(list)
        for _, dirs, _ in os.walk(self.masks_path):
            for dir in dirs:
                for _, _, filenames in os.walk(os.path.join(self.masks_path, dir)):
                    for filename in filenames:
                        if filename.startswith(".") or ".db" in filename:
                            continue
                        mask_name_to_image_paths[filename].append(os.path.join(self.masks_path, dir, filename))
        os.makedirs(os.path.join(self.source_path, self.mask_folder_name), exist_ok=True)
        for filename, paths in mask_name_to_image_paths.items():
            joined_mask_image = None
            for path in paths:
                if joined_mask_image is None:
                    joined_mask_image = cv2.imread(path)
                else:
                    joined_mask_image = cv2.bitwise_or(joined_mask_image, cv2.imread(path))
            if self.img_prefix not in filename:
                new_filename = f"{self.segmentation_prefix}_{filename}"
            else:
                new_filename = filename.replace(self.img_prefix, self.segmentation_prefix)
            if not os.path.isfile(os.path.join(self.source_path, self.mask_folder_name, new_filename)):
                cv2.imwrite(os.path.join(self.source_path, self.mask_folder_name, new_filename), joined_mask_image)

        return X
