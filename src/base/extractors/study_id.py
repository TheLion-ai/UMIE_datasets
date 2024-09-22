"""
This module contains the definition of the BaseStudyIdExtractor class.

The BaseStudyIdExtractor class is a base class for study ID extraction. It provides a default implementation
for extracting the study ID from a given path.
Each medical imaging examination contains many imgs, study id identifies all the imgs from the same examination.
"""

import os


class BaseStudyIdExtractor:
    """
    Base class for study ID extractors.

    This class provides a base implementation for extracting study IDs from file paths.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, path: str) -> str:
        """
        Call the _extract method to extract the study id from the given path.

        Args:
            path (str): The path from which to extract the study id.

        Returns:
            str: The extracted study id.
        """
        return self._extract(path)

    def _extract(self, path: str) -> str:
        """
        Extract the study ID from the given path.

        Args:
            path (str): The path from which to extract the study ID.

        Returns:
            str: The extracted study ID.
        """
        return path
