"""Create file tree for dataset."""

import json
import os
from typing import Any

from umie_datasets.base.step import BaseStep


class CreateFileTree(BaseStep):
    """Create file tree for dataset."""

    def transform(self, X: list) -> list:
        """Create file tree for dataset.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        self.create_file_tree()
        return X

    def _create_dir(self, *filetree: Any) -> None:
        """Create directory if it does not exist.

        Args:
            *filetree (list): List of folder paths.
        """
        filetree_path = os.path.join(*filetree)
        if not os.path.exists(filetree_path):
            os.makedirs(filetree_path)

    def _create_json(self) -> None:
        """Create JSON file for storing information about target datafiles."""
        # Always create a new file
        with open(self.json_path, "w"):
            pass

    def create_file_tree(self) -> None:
        """Create file tree for dataset."""
        self._create_dir(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        self._create_json()

        for phase in self.phases.values():
            self._create_dir(self.target_path, f"{self.dataset_uid}_{self.dataset_name}", phase)

            self._create_dir(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase,
                self.image_folder_name,
            )

            if self.masks_path:
                self._create_dir(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase,
                    self.mask_folder_name,
                )
