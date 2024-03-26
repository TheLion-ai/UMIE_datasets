"""Delete temporary files."""

import csv
import os
import tempfile
from typing import Callable

import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class DeleteTempFiles(TransformerMixin):
    """Delete temporary files."""

    def __init__(self, target_path: str, **kwargs: dict):
        """Delete temporary files.

        Args:
            target_path (str): Path to the target directory.
        """
        self.target_path = target_path

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Delete temporary files.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        source_paths_file = os.path.join(self.target_path, "source_paths.csv")
        if os.path.exists(source_paths_file):
            os.remove(source_paths_file)

        return X
