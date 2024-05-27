"""Converts jpg and jpeg files to png images."""
import glob
import os
from typing import Callable

import PIL
from sklearn.base import TransformerMixin
from tqdm import tqdm


class ConvertJpg2Png(TransformerMixin):
    """Converts jpg files to png images."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str,
        mask_folder_name: str,
        masks_path: str = None,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        phase_extractor: Callable = lambda x: x,
        img_prefix: str = "",
        **kwargs: dict,
    ):
        """Convert jpg files to png images.

        Args:
            source_path (str): Path to source folder.
            target_path (str): Path to the target folder.
            masks_path (str): Path to the source folder with masks.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            img_prefix (str, optional): Prefix for the file with images.
        """
        self.source_path = source_path
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.phase_extractor = phase_extractor
        self.img_prefix = img_prefix

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
        for img_path in tqdm(X):
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
            image = PIL.Image.open(img_path)
            image.save(new_path, format="png")
            if self.mask_folder_name and self.mask_folder_name in img_path:
                os.remove(img_path)
        except IOError:
            print("Image not found")
