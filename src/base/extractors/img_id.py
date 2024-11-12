"""
This module contains the definition of the BaseImgIdExtractor class.

BaseImgIdExtractor is an abstract base class for extracting image IDs from file paths.
you must either create a subclass of BaseImgIdExtractor and override the _extract method or use DefaultImgIdExtractor.
"""

import os
from abc import ABC
from pathlib import Path

from .base_id import BaseIdExtractor


class BaseImgIdExtractor(BaseIdExtractor, ABC):
    """Abstract base class for extracting image IDs from file paths."""

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


class DefaultImgIdExtractor(BaseImgIdExtractor):
    """
    This class provides a default implementation for extracting the img ID from a given path.

    Attributes:
        None

    Methods:
        _extract(self, path: str) -> str: Default implementation of the img ID extraction.

    """

    def _extract(self, img_path: str) -> str:
        """Use the default method to extract the image ID from the path.

        Args:
            path (str): The file path from which to extract the image ID.

        Returns:
            str: The extracted image ID.
        """
        # by default, extract the image ID from the file name
        return Path(img_path).name
