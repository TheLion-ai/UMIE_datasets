"""Converts jpg and jpeg files to png images."""
import glob
import os
from typing import Callable

import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
from sklearn.base import TransformerMixin
from tqdm import tqdm


class ConvertJpg2Png(TransformerMixin):
    """Converts jpg files to png images."""

    def __init__(
        self,
        target_path: str,
        masks_path: str,
        dataset_name: str,
        dataset_uid: str,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        **kwargs: dict,
    ):
        """Convert jpg files to png images.

        Args:
            target_path (str): Path to the target folder.
            masks_path (str): Path to the source folder with masks.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
        """
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor

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
            if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):
                self.convert_tif2png(img_path)

        root_path = os.path.join(
            os.path.dirname(os.path.dirname(X[0])), self.mask_folder_name
        )
        mask_paths = glob.glob(f"{root_path}/**/*.jpg", recursive=True) + glob.glob(
            f"{root_path}/**/*.jpeg", recursive=True
        )
        if mask_paths:
            for mask_path in mask_paths:
                self.convert_tif2png(mask_path)
        else:
            print("Masks not found.")

        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(f"{root_path}/*.png", recursive=True)
        return new_paths

    def convert_tif2png(self, img_path: str) -> None:
        """Convert tif files to png images.

        Args:
            img_path (str): Path to the image.
        """
        png_path = img_path.replace(".jpg", ".png").replace(".jpeg", ".png")
        try:
            image = PIL.Image.open(img_path)
            image.save(png_path, format="png")
            os.remove(img_path)
        except IOError:
            print("Image not found")
