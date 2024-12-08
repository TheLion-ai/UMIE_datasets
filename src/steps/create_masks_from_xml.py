"""Converts masks from xml files to png images with appropriate color encoding."""

import glob
import json
import os
import plistlib
import re

import cv2
import numpy as np
from tqdm import tqdm

from base.step import BaseStep


class CreateMasksFromXml(BaseStep):
    """Converts masks from xml files to png images with appropriate color encoding."""

    def transform(self, X: list) -> list:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        print("Creating masks from xml files...")
        mask_paths = glob.glob(f"{self.masks_path}/**/*.xml", recursive=True)
        # TODO Print warning if no xml paths available in path / path empty.
        # Make prompt whether to use source_path instead, otherwise skip this step,
        # or whole pipeline. If we skip step, all converted images will be wiped.

        file_to_dicom_attr_mapping_file = os.path.join(self.target_path, "file_to_dcm_attribute_mapping.json")
        self.file_to_dicom_attr_mapping = None
        if os.path.exists(file_to_dicom_attr_mapping_file):
            self.file_to_dicom_attr_mapping = json.load(open(file_to_dicom_attr_mapping_file))

        for mask_path in tqdm(mask_paths):
            self.xml_mask_creator(mask_path)

        # Free the memory occupied by whole json file as it will not be used anymore
        self.file_to_dicom_attr_mapping = None

        return X
