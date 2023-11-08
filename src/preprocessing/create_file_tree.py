"""Create file tree for dataset."""
import glob
import os
import re
import shutil

import cv2
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


class CreateFileTree(BaseEstimator, TransformerMixin):
    """Create file tree for dataset."""
    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        **kwargs,
    ):
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """Create file tree for dataset.
        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        self.create_file_tree()
        return X

    def _create_dir(self, *filetree):
        """Create directory if it does not exist.
        Args:
            *filetree (list): List of folder paths.
        """
        filetree_path = os.path.join(*filetree)
        if not os.path.exists(filetree_path):
            os.makedirs(filetree_path)

    def create_file_tree(self):
        """Create file tree for dataset."""
        self._create_dir(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")

        if len(self.phases.keys()) > 1:
            for phase in self.phases.values():
                self._create_dir(
                    self.target_path, f"{self.dataset_uid}_{self.dataset_name}", phase
                )
                self._create_dir(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase,
                    self.image_folder_name,
                )
                self._create_dir(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase,
                    self.mask_folder_name,
                )

        else:
            self._create_dir(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.image_folder_name,
            )
            self._create_dir(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.mask_folder_name,
            )
