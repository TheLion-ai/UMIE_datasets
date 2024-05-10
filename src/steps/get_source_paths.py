"""Temporary store paths to images in source directory and their new names."""
import csv
import os
import tempfile
from typing import Callable

import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class GetSourcePaths(TransformerMixin):
    """Temporary store paths to images in source directory and their new names."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: os.path.basename(x),
        phase_extractor: Callable = lambda x: os.path.basename(x),
        **kwargs: dict,
    ):
        """Temporary store paths to images in source directory and their new names.

        Args:
            source_path (str): Path to the source directory.
        """
        self.source_path = source_path
        self.target_path = target_path
        self.omit_conditions = list
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.paths_dict: dict[str, str] = {}

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Temporary store paths to images in source directory and their new names.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        for img_path in tqdm(X):
            self.get_source_paths(img_path)

        with open(os.path.join(self.target_path, "source_paths.csv"), "w", newline="") as temp_file:
            writer = csv.writer(temp_file)
            paths_data = np.transpose(np.vstack([list(self.paths_dict.keys()), list(self.paths_dict.values())]), (1, 0))
            writer.writerows(list(paths_data))

        return X

    def get_source_paths(self, img_path: str) -> None:
        """Temporary store paths to images in source directory and their new names.

        Args:
            source_path (str): Path to the source directory.
        Returns:
            file_paths (list): List of paths to the images.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_extractor(img_path)
        # Save source path using temporary image id, which consists of phase_id, study_id and img_id
        self.paths_dict[f"{phase_id}_{study_id}_{img_id}"] = img_path
