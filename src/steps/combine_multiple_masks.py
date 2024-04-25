"""Combine and recolor multiple masks into one."""

import glob
import os

import cv2
import numpy as np
import PIL.Image
from sklearn.base import TransformerMixin
from tqdm import tqdm


class CombineMultipleMasks(TransformerMixin):
    """Combine and recolor multiple masks into one."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        dataset_masks: dict,
        mask_encodings: dict,
        mask_colors_source2target: dict,
        mask_selector: str,
        multiple_masks_selector: dict,
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
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.dataset_masks = dataset_masks
        self.mask_encodings = mask_encodings
        self.mask_colors_source2target = mask_colors_source2target
        self.mask_selector = mask_selector
        self.multiple_masks_selector = multiple_masks_selector
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
        # Robust to multiple modalities and lack of masks for some images
        # root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        # mask_paths = glob.glob(os.path.join(root_path, f"**/{self.mask_folder_name}/*.png"), recursive=True)
        # print("Recoloring masks...")
        # for mask_path in mask_paths:
        #     if os.path.exists(mask_path):
        #         self.combine_multiple_masks(mask_path)

        for root, _, filenames in os.walk(self.source_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    path = os.path.join(root, filename)
                    # Verify if file is not a mask
                    if self.mask_selector in path:
                        self.combine_multiple_masks(path)

        return X

    def combine_multiple_masks(self, mask_path: str) -> None:
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_path (str): Path to the mask.
        """
        # changing pixel values
        active_selector = [k for k in self.multiple_masks_selector.keys() if k in mask_path]
        if bool(active_selector):
            new_mask_path = mask_path.replace(active_selector[0], "").replace("__", "_")
            # new_mask_path = ""
            if os.path.exists(new_mask_path):
                return
            mask = cv2.imread(mask_path)
            source_color = self.dataset_masks[self.multiple_masks_selector[active_selector[0]]]
            target_color = self.mask_encodings[self.multiple_masks_selector[active_selector[0]]]
            np.place(mask, mask == source_color, target_color)
            for k, v in self.multiple_masks_selector.items():
                other_mask_path = mask_path.replace(active_selector[0], k)
                # other_mask_path = ""
                other_mask = cv2.imread(other_mask_path)
                source_color = self.dataset_masks[v]
                target_color = self.mask_encodings[v]
                np.place(other_mask, other_mask == source_color, target_color)
                mask[other_mask > 0] = other_mask[other_mask > 0]

            cv2.imwrite(new_mask_path, mask)
