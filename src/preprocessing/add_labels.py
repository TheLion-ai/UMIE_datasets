"""Add labels to the images and masks based on the labels.json file."""
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class AddLabels(BaseEstimator, TransformerMixin):
    """Add labels to the images and masks based on the labels.json file."""
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
        labels_path: str = "",
        get_label: callable = lambda x: [],
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
        self.labels_path = (labels_path,)
        self.get_label = get_label

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X,  # img_paths
    ):
        """Add labels to the images and masks.
        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        labels_path_extention = os.path.basename(self.labels_path).split(".")[1]
        if labels_path_extention == "json":
            with open(self.labels_path) as f:
                labels_list = json.load(f)

        print("Adding labels...")
        for img_path in tqdm(X):
            self.add_labels(img_path, labels_list)
        root_path = os.path.dirname(X[0])
        # TODO: Add f"{root_path}/**/*.png" to all new paths
        new_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        return new_paths

    def add_labels(self, img_path, labels_list):
        """Add labels to the image and mask.
        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        root_path = os.path.dirname(img_path)
        img_id = os.path.basename(img_path).split(".")[0]
        label_prefix = "-"
        labels = self.get_labels(img_path)
        labels_str = "".join([label_prefix + label for label in labels])
        new_name = f"{img_id}{labels_str}.png"
        os.rename(img_path, os.path.join(root_path, new_name))

        root_path = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(root_path, self.mask_folder_name, img_id)
        if os.path.exists(mask_path):
            os.rename(
                mask_path,
                f"{root_path}/{self.mask_folder_name}/{img_id}{labels_str}.png",
            )
