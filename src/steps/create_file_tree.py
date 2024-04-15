"""Create file tree for dataset."""

import os
from typing import Any

from sklearn.base import TransformerMixin


class CreateFileTree(TransformerMixin):
    """Create file tree for dataset."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str | None,
        mask_folder_name: str | None,
        **kwargs: dict,
    ):
        """Create file tree for dataset.

        Args:
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str | None, optional): Name of the folder with images.
                                                      Defaults to "Images". If None,
                                                      folder is not created.
            mask_folder_name (str | None, optional): Name of the folder with masks.
                                                     Defaults to "Masks". If None,
                                                     folder is not created.
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name

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

    def create_file_tree(self) -> None:
        """Create file tree for dataset."""
        self._create_dir(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")

        for phase in self.phases.values():
            self._create_dir(self.target_path, f"{self.dataset_uid}_{self.dataset_name}", phase)

            if self.image_folder_name:
                self._create_dir(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase,
                    self.image_folder_name,
                )

            if self.mask_folder_name:
                self._create_dir(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase,
                    self.mask_folder_name,
                )
