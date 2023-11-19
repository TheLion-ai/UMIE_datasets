"""Get file paths of all images from a source directory."""
import os

from sklearn.base import TransformerMixin


class GetFilePaths(TransformerMixin):
    """Get file paths of all images from a source directory."""

    def __init__(self, source_path: str = "", **kwargs: dict):
        """Get file paths of all images from a source directory.

        Args:
            source_path (str): Path to the source directory.
        """
        self.source_path = source_path
        self.omit_conditions = list

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
        for root, dirnames, filenames in os.walk(source_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    file_paths.append(os.path.join(root, filename))
        print(file_paths)
        return file_paths
