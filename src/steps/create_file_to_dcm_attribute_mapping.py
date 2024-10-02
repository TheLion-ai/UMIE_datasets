"""Create filename to dicom attribute mapping."""

import json
import os
from typing import Any

from pydicom import dcmread

from base.step import BaseStep


class CreateFileToDcmAttributeMapping(BaseStep):
    """Create filename to dicom attribute mapping."""

    def transform(self, X: list) -> list:
        """Create filename to dicom attribute mapping.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if self.dicom_mapping_attribute:
            self.create_file_to_dcm_attribute_mapping(X)
        else:
            raise ValueError("No Dicom Attribute to map to provided.")

        return X

    def create_file_to_dcm_attribute_mapping(self, path_list: list) -> None:
        """Create filename to dicom attribute mapping.

        Args:
            path_list (list): List of paths to the images.
        """
        file_to_dicom_attr_mapping = dict()
        for filepath in path_list:
            if filepath.endswith(".dcm"):
                ds = dcmread(filepath)
                file_to_dicom_attr_mapping[getattr(ds, self.dicom_mapping_attribute)] = filepath

        with open(os.path.join(self.target_path, "file_to_dcm_attribute_mapping.json"), "w") as temp_file:
            temp_file.write(json.dumps(file_to_dicom_attr_mapping))
