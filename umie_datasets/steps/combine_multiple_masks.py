"""Combine and recolor multiple masks into one."""

import glob
import os

import cv2
import numpy as np
import PIL.Image
from tqdm import tqdm

from umie_datasets.base.step import BaseStep


class CombineMultipleMasks(BaseStep):
    """Combine and recolor multiple masks into one."""

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if len(X) == 0:
            raise ValueError("No list of files provided.")

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
            if os.path.exists(new_mask_path):
                return
            mask = cv2.imread(mask_path)
            mask_name = self.multiple_masks_selector[active_selector[0]]
            source_color = self.masks[mask_name]["source_color"]
            target_color = self.mask_encodings[mask_name]["target_color"]
            np.place(mask, mask == source_color, target_color)
            for k, v in self.multiple_masks_selector.items():
                other_mask_path = mask_path.replace(active_selector[0], k)
                # other_mask_path = ""
                other_mask = cv2.imread(other_mask_path)
                source_color = self.masks[v]["source_color"]
                target_color = self.masks[v]["target_color"]
                np.place(other_mask, other_mask == source_color, target_color)
                mask[other_mask > 0] = other_mask[other_mask > 0]

            cv2.imwrite(new_mask_path, mask)
