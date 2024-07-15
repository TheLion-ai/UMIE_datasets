"""Converts jpg and jpeg files to png images."""
import glob
import os
from typing import Callable

from PIL import Image
from tqdm import tqdm

from base.extractors.img_id import BaseImgIdExtractor
from base.step import BaseStep


class ConvertJpg2Png(BaseStep):
    """Converts jpg files to png images."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert jpg files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Converting jpg to png...")
        paths = glob.glob(os.path.join(self.source_path, "**/*.*"), recursive=True)
        for img_path in tqdm(paths):
            if img_path.endswith(".jpg") or img_path.endswith(".jpeg") or img_path.endswith(".JPG"):
                # Convert each jpg or jpeg file to png
                self.convert_jpg2png(img_path)

        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        mask_paths = glob.glob(
            os.path.join(root_path, f"**/{self.mask_folder_name}/*.jpg"), recursive=True
        ) + glob.glob(os.path.join(root_path, f"**/{self.mask_folder_name}/*.jpeg"), recursive=True)
        if mask_paths:
            for mask_path in tqdm(mask_paths):
                self.convert_jpg2png(mask_path)
        else:
            print("Masks not found.")

        # Get paths to all images after conversion
        new_paths = glob.glob(os.path.join(self.source_path, f"**/*{self.img_prefix}*png"), recursive=True)
        return new_paths

    def convert_jpg2png(self, img_path: str) -> None:
        """Convert jpg to png images.

        Args:
            img_path (str): Path to the image.
        """
        # Path for image after conversion
        new_path = img_path.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".JPG", ".png")
        try:
            # Open image, save with new extension and remove old file
            image = Image.open(img_path)
            image.save(new_path, format="png")
            if self.mask_folder_name and self.mask_folder_name in img_path:
                os.remove(img_path)
        except IOError:
            print("Image not found")
