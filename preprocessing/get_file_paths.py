import os
import glob
import cv2
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


class GetFilePaths(BaseEstimator, TransformerMixin):

    def __init__(self, source_path: str="", **kwargs):
        self.source_path = source_path
        self.omit_conditions = list

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X="",
    ):
        file_paths = self.get_file_paths(X)
        return file_paths


    def get_file_paths(self, source_path: str):
        file_paths = []
        for root, dirnames, filenames in os.walk(source_path):
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                else:
                    file_paths.append(os.path.join(root, filename))
        return file_paths
