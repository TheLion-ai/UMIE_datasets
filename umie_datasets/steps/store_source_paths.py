"""Temporary store paths to images in source directory and their new names."""
import json
import os

from umie_datasets.base.extractors.img_id import BaseImgIdExtractor
from umie_datasets.base.step import BaseStep


class StoreSourcePaths(BaseStep):
    """Temporary store paths to images in source directory and their new names."""

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
        source_paths_dict = {self.get_umie_img_path(img_path): img_path for img_path in X}  # {umie_id: src_img_path}

        with open(os.path.join(self.target_path, "source_paths.json"), "w") as temp_file:
            temp_file.write(json.dumps(source_paths_dict))

        return X
