"""This module contains the definition of the BaseImgIdExtractor class, which is an abstract base class for extracting image IDs from file paths."""

import os
from abc import ABC, abstractmethod


class BaseImgIdExtractor(ABC):
    """Abstract base class for extracting image IDs from file paths."""

    def __call__(self, path: str) -> str:
        """Extract image id from the path.

        Args:
            path (str): The file path from which to extract the image ID.

        Returns:
            str: The extracted image ID.

        Examples:
            >>> extractor = BaseImgIdExtractor()
            >>> extractor('/path/to/image_001.jpg')
            'image_001.jpg'
        """
        return self._extract(path)

    def _extract(self, path: str) -> str:
        """Use the default method to extract the image ID from the path.

        Args:
            path (str): The file path from which to extract the image ID.

        Returns:
            str: The extracted image ID.
        """
        return os.path.basename(path)
