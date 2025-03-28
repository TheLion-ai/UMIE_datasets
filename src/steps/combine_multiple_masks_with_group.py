"""Combine and recolor multiple masks into one with grouping function."""
import os
from typing import Dict, List

import cv2
import numpy as np

from base.step import BaseStep

masks_dict: Dict[str, List[str]] = {}


class CombineMultipleMasksWithGroup(BaseStep):
    """Combine and recolor multiple masks into one with grouping function."""

    # change to define in pipeline #TODO
    def group_masks(self, mask_path: str) -> None:
        """Group masks of same picture."""
        mask_id = self.study_id_extractor(mask_path)[:-2]
        if mask_id not in masks_dict:
            masks_dict[mask_id] = []
        masks_dict[mask_id].append(mask_path)

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
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
                    if self.mask_selector(path):
                        if path.endswith(".png"):
                            self.group_masks(path)
        self.combine_masks()

        return X

    def combine_masks(self) -> None:
        """Recolors masks from default color to the color specified in the config and combine."""
        for mask_id in masks_dict:
            masks = masks_dict[mask_id]
            mask_path = masks[0]
            combined_mask = None
            for mask in masks:
                mask_img = cv2.imread(mask)
                for key in self.multiple_masks_selector.keys():
                    if key in mask:
                        mask_name = self.multiple_masks_selector[key]
                        source_color = self.masks[mask_name]["source_color"]
                        target_color = self.masks[mask_name]["target_color"]
                np.place(mask_img, mask_img == source_color, target_color)
                if combined_mask is None:
                    combined_mask = mask_img
                else:
                    combined_mask[mask_img > 0] = mask_img[mask_img > 0]
            cv2.imwrite(mask_path, combined_mask)
