"""
This module contains the definition of the BaseImgIdExtractor class.

BaseImgIdExtractor is an abstract base class for extracting image IDs from file paths.
If you want to modify the default behaviour of the image ID extraction, you can create a subclass of BaseImgIdExtractor and override the _extract method.
"""

import os
from abc import ABC, abstractmethod


class BaseImgIdExtractor(ABC):
    """Abstract base class for extracting image IDs from file paths."""

    def __call__(self, path: str) -> str:
        """Extract image id from the path.

        Args:
            path (str): The file path from which to extract the image ID. Usually the source path.

        Returns:
            str: The extracted image ID.

        Examples:
            >>> extractor = BaseImgIdExtractor()
            >>> extractor('/path/to/image_001.png')
            'image_001.png'
        """
        return self._extract(path)

    def _extract(self, path: str) -> str:
        """Use the default method to extract the image ID from the path.

        Args:
            path (str): The file path from which to extract the image ID.

        Returns:
            str: The extracted image ID.
        """
        # by default, extract the image ID from the file name
        return os.path.basename(path)
