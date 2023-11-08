"""Change image ids to match the format of the rest of the dataset."""
import glob
import os
import re
import shutil

import cv2
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class AddNewIds(BaseEstimator, TransformerMixin):
    """Change img ids to match the format of the rest of the dataset."""
    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        img_id_extractor: callable = lambda x: os.path.basename(x),
        study_id_extractor: callable = lambda x: x,
        phase_extractor: callable = lambda x: x,
        mask_selector: str = "segmentations",
        **kwargs,
    ):
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.mask_selector = mask_selector

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X,  # img_paths
    ):
        """Change image ids to match the format of the rest of the dataset.
        Args:
            X (list): List of paths to the images.
        Returns:
            new_paths (list): List of paths to the images with new ids.
        """
        print("Adding new ids to the dataset...")
        for img_path in tqdm(X):
            self.add_new_ids(img_path)

        root_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            self.image_folder_name,
        )
        new_paths = glob.glob(f"{root_path}/*.*", recursive=True)
        return new_paths

    def add_new_ids(self, img_path):
        """Change image ids to match the format of the rest of the dataset.
        Args:
            img_path (str): Path to the image.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)

        if len(self.phases.keys()) <= 1:
            new_file_name = f"{self.dataset_uid}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.image_folder_name,
                new_file_name,
            )
        else:
            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            elif self.mask_selector in img_path:
                return None
            phase_name = self.phases[phase_id]
            new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.image_folder_name,
                new_file_name,
            )

        if not os.path.exists(new_path):
            shutil.copy2(img_path, new_path)
