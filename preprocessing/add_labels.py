"""Add labels to filenames."""
import os
import glob

import yaml
from sklearn.base import BaseEstimator, TransformerMixin

class AddLabelsToFilenames((BaseEstimator, TransformerMixin):):

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(
        self,
        X: str=None,
        y: str=None,
        source_path: str="",
        target_path: str="",
        dataset_name: str="",
        label: str=""
    ):
    """
    Add labels to filenames.

    Args:
        source_path (str): path to the root folder with masks and images
        target_path (str): path to the folder where recolored masks will be saved
        dataset_name (str): name of the dataset to be renamed, check dataset config for the list of available uids for datasets and add new ones if needed
        label (str): label to be added to the filenames
    """
    # TODO: add specific functions for the dataset
    pass
