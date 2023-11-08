"""Converts masks from xml files to png images with appropriate color encoding."""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class CreateBlankMasks(BaseEstimator, TransformerMixin):

    def __init__(self, mask_folder_name: str="Masks", **kwargs):
        self.mask_folder_name = mask_folder_name
        

    def fit(self, X, y=None):
            return self

    def transform(
            self,
            X: str,
            target_path: str="",
        ):
        mask_paths = glob.glob(f"{os.path.join(target_path, self.mask_folder_name)}/**/*.png", recursive=True)
        mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Creating blank masks...")
        for img_path in tqdm(X):
            img_name = os.path.basename(img_path)
            if img_name not in mask_names:
                self.create_blank_masks(img_path)
        return X



    def create_blank_masks(self, img_path: str):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        new_path = os.path.dirname(os.path.dirname(img_path))
        new_path = os.path.join(new_path, self.mask_folder_name, img_name)
        mask = np.zeros(img.shape, np.uint8)
        cv2.imwrite(new_path, mask)
