"""Converts tif files to png images."""
import glob
import os
import shutil
from typing import Callable

import numpy as np
import PIL
from sklearn.base import TransformerMixin
from tqdm import tqdm


class ConvertTif2Png(TransformerMixin):
    """Converts tif files to png images."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        masks_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str,
        mask_folder_name: str,
        mask_selector: str,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        phase_extractor: Callable = lambda x: x,
        **kwargs: dict,
    ):
        """Convert tif files to png images.

        Args:
            source_path (str): Path to the source folder.
            target_path (str): Path to the target folder.
            masks_path (str): Path to the source folder with masks.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            mask_selector (str): Phrase included only in masks paths.
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
        """
        self.source_path = source_path
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.mask_selector = mask_selector
        self.img_id_extractor = img_id_extractor
        self.phase_extractor = phase_extractor

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert tif files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Converting tif to png...")
        for img_path in tqdm(X):
            if img_path.endswith(".tif") or img_path.endswith(".tiff"):
                self.convert_tif2png(img_path)

        mask_paths = []
        for root, _, filenames in os.walk(self.masks_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    path = os.path.join(root, filename)
                    # Verify if file is not a mask
                    if self.mask_selector in path:
                        mask_paths.append(path)
        if mask_paths:
            for mask_path in mask_paths:
                self.convert_tif2png(mask_path)
        else:
            print("Masks not found.")

        new_paths = glob.glob(os.path.join(self.source_path, "**/*.png"), recursive=True)
        return new_paths

    def convert_tif2png(self, img_path: str) -> None:
        """Convert tif files to png images.

        Args:
            img_path (str): Path to the image.
        """
        if self.mask_folder_name not in img_path:

            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            phase_name = self.phases[phase_id]
            # Changing .tif to .tiff, so images will be readable for PIL
            if self.img_id_extractor(img_path).endswith(".tif"):
                png_filename = self.img_id_extractor(img_path).replace(".tif", ".tiff")
            else:
                png_filename = self.img_id_extractor(img_path)
            # Copy tiff image to target directory
            tiff_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.image_folder_name,
                png_filename,
            )
            shutil.copy2(img_path, tiff_path)

            new_path = img_path.replace(".tif", ".png").replace(".tiff", ".png")
        else:
            new_path = img_path.replace(".tif", ".png").replace(".tiff", ".png")
            tiff_path = img_path
        # Image conversion
        try:
            image = PIL.Image.open(tiff_path)
            invert = True if np.min(np.array(image)) < 0 else False
            image = image.convert("L")
            if invert:
                image = PIL.ImageOps.invert(image)
            image.save(new_path, format="png")
            if self.target_path in tiff_path:
                os.remove(tiff_path)
        except IOError:
            print("Image not found")
