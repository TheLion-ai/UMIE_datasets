import glob
import os
import re
import shutil

import cv2
import nibabel as nib
import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class ConvertNii2Png(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        window_center: int,
        window_width: int,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        img_id_extractor: callable = lambda x: os.path.basename(x),
        study_id_extractor: callable = lambda x: x,
        phase_extractor: callable = lambda x: x,
        zfill: int = 3,
        **kwargs,
    ):
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

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X,  # img_paths
    ):
        print("Converting nii to png...")
        for img_path in tqdm(X):
            if img_path.endswith(".nii.gz"):
                self.convert_nii2png(img_path)
        root_path = os.path.dirname(X[0])
        # TODO: Add f"{root_path}/**/*.png" to all new paths
        new_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        return new_paths

    def convert_nii2png(self, img_path):
        nii_img = nib.load(img_path)
        nii_data = nii_img.get_fdata()
        slices = nii_data.shape[0]
        for idx in range(slices):
            root_path = os.path.dirname(img_path)
            name = (
                os.path.basename(img_path).split(".")[0]
                + f"_{str(idx).zfill(self.zfill)}.png"
            )
            new_path = os.path.join(root_path, name)
            # TODO: remove hardcoded names
            img = np.array(nii_data[idx, :, :])
            if "segmentation" not in new_path:
                img = self._apply_window(img)
            cv2.imwrite(new_path, img)

    def _apply_window(self, pixel_data):
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
