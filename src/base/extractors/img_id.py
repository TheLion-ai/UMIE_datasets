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

    def _return_zero(self) -> str:
        """Return "0.png" as image ID.

        Returns:
            str: "0.png".
        """
        return "0.png"

    def _extract_by_separator(self, path: str, separator: str, index: int = -1, maxsplit: int = -1) -> str:
        """Extract image ID from the path by the separator.

        Args:
            path (str): The file path from which to extract the image ID.
            separator (str): The separator to split the path.
            index (int): The index of the image ID in the split path. Defaults to -1.
            maxsplit (int): The maximum number of splits to perform. Defaults to -1.

        Returns:
            str: The extracted image ID.
        """
        return os.path.basename(path).rsplit(separator, maxsplit=maxsplit)[index]

    def _extract_first_two_chars(self, path: str, extract_zeros: bool = False) -> str:
        """Extract first two characters of filename from the path.

        Args:
            path (str): The file path from which to extract the image ID.
            extract_zeros (bool): If True, extract "00" as image ID. Defaults to False.

        Returns:
            str: The extracted image ID.
        """
        ext = os.path.splitext(os.path.basename(path))[1]
        if extract_zeros:
            img_id = "00"
        else:
            img_id = os.path.basename(path)[:2]
        return img_id + ext
