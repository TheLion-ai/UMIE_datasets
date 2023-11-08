"""Get file paths of all images from a source directory."""
import glob
import os

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GetFilePaths(BaseEstimator, TransformerMixin):
    """Get file paths of all images from a source directory."""
    def __init__(self, source_path: str = "", **kwargs):
        self.source_path = source_path
        self.omit_conditions = list

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X="",
    ):
        """Get file paths of all images from a source directory.
        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        file_paths = self.get_file_paths(X)
        return file_paths

    def get_file_paths(self, source_path: str):
        """Get file paths of all images from a source directory.
        Args:
            source_path (str): Path to the source directory.
        Returns:
            file_paths (list): List of paths to the images.
        """
        file_paths = []
        for root, dirnames, filenames in os.walk(source_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    file_paths.append(os.path.join(root, filename))
        return file_paths
