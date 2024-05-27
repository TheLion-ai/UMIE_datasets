"""Converts nii files to png images with appropriate color encoding."""

import glob
import os
from typing import Callable

import cv2
import nibabel as nib
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class ConvertNii2Png(TransformerMixin):
    """Converts nii files to png images with appropriate color encoding."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        window_center: int,
        window_width: int,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        zfill: int = 3,
        img_prefix: str = "imaging",
        segmentation_prefix: str = "segmentation",
        **kwargs: dict,
    ):
        """Convert nii files to png images with appropriate color encoding.

        Args:
            source_path (str): Path to source folder.
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            window_center (int): Window center for the images.
            window_width (int): Window width for the images.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 3.
            img_dicom_prefix (str, optional): Prefix for the file with images. Defaults to "imaging".
            segmentation_dicom_prefix (str, optional): Prefix for the dicom file with segmentations. Defaults to "segmentation".
        """
        self.source_path = source_path
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.window_center = window_center
        self.window_width = window_width
        self.zfill = zfill
        self.img_dcm_prefix = img_prefix
        self.segmentation_dcm_prefix = segmentation_prefix

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert nii files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Converting nii to png...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            if img_path.endswith(".nii.gz"):
                if self.segmentation_dcm_prefix in img_path or self.img_dcm_prefix in img_path:
                    self.convert_nii2png(img_path)
        new_paths = glob.glob(os.path.join(self.source_path, f"**/{self.img_dcm_prefix}*.png"), recursive=True)
        return new_paths

    def convert_nii2png(self, img_path: str) -> None:
        """Convert nii files to png images with appropriate color encoding.

        Args:
            img_path (str): Path to the image.
        """
        nii_img = nib.load(img_path)
        nii_data = nii_img.get_fdata()
        slices = nii_data.shape[0]
        for idx in range(slices):
            root_path = os.path.dirname(img_path)
            name = os.path.basename(img_path).split(".")[0] + f"_{str(idx).zfill(self.zfill)}.png"
            new_path = os.path.join(root_path, name)
            img = np.array(nii_data[idx, :, :])
            if self.segmentation_dcm_prefix not in new_path:
                img = self._apply_window(img)

            cv2.imwrite(new_path, img)

    def _apply_window(self, pixel_data: np.ndarray) -> np.ndarray:
        """Apply window to the image.

        Args:
            pixel_data (np.ndarray): Image data.
        Returns:
            np.ndarray: Image data with applied window.
        """
        # apply window
        pixel_data = np.clip(
            pixel_data,
            self.window_center - self.window_width / 2,
            self.window_center + self.window_width / 2,
        )
        # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
        min = np.min(pixel_data)
        min = -1000 if min < -1000 else min
        pixel_data = pixel_data - min
        ratio = np.max(pixel_data) / 255
        pixel_data = np.divide(pixel_data, ratio).astype(int)
        return pixel_data
