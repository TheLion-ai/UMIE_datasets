"""Converts tif files to png images."""
import glob
import os
import shutil
from typing import Callable

import numpy as np
import PIL
from tqdm import tqdm

from umie_datasets.base.extractors.img_id import BaseImgIdExtractor
from umie_datasets.base.step import BaseStep


class ConvertTif2Png(BaseStep):
    """Converts tif files to png images."""

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

            phase_id = self.phase_id_extractor(img_path)
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
