"""Create blank masks for images that don't have masks."""
import glob
import os

import cv2
import jsonlines
import numpy as np
from tqdm import tqdm

from base.step import BaseStep


class CreateBlankMasks(BaseStep):
    """Create blank masks for images that don't have masks."""

    def transform(self, X: list) -> list:
        """Create blank masks for images that don't have masks.

        Args:
            X (list): List of paths to the images.
            target_path (str): Path to the target folder.
        Returns:
            X (list): List of paths to the images.
        """
        mask_paths = glob.glob(
            os.path.join(
                self.target_path, f"**/{self.dataset_uid}_{self.dataset_name}/**/{self.mask_folder_name}/*.png"
            ),
            recursive=True,
        )

        mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Creating blank masks...")
        self.blank_masks: list = []
        for img_path in tqdm(X):
            img_name = os.path.basename(img_path)
            if img_name not in mask_names:
                self.create_blank_masks(img_path)

        updated_lines = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                if obj["umie_path"] in self.blank_masks:
                    mask_path = self.get_umie_mask_path(obj["umie_path"])
                    obj["mask_path"] = self.get_path_without_target_path(mask_path)

                updated_lines.append(obj)

        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in updated_lines:
                writer.write(obj)

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

        umie_path = new_path.replace(self.mask_folder_name, self.image_folder_name)
        self.blank_masks.append(self.get_path_without_target_path(umie_path))
