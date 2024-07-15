"""Get file paths of all images from a source directory."""

import os
from typing import Optional

from base.step import BaseStep


class GetFilePaths(BaseStep):
    """Get file paths of all images from a source directory."""

    def transform(
        self,
        X: list = [],
    ) -> list:
        """Get file paths of all images from a source directory.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        file_paths = self.get_file_paths(self.source_path)
        return file_paths

    def get_file_paths(self, source_path: str) -> list:
        """Get file paths of all images from a source directory.

        Args:
            source_path (str): Path to the source directory.
        Returns:
            file_paths (list): List of paths to the images.
        """
        file_paths = []
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if filename.endswith((".csv", ".json", ".xslx", ".yml", ".txt")):
                    continue
                if filename.startswith("LICENSE"):
                    continue
                else:
                    path = os.path.join(root, filename)
                    # Verify if file is not a mask
                    if self.mask_selector is None or self.mask_selector not in path:
                        file_paths.append(path)
        return file_paths
