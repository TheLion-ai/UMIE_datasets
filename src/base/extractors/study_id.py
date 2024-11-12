"""
This module contains the definition of the BaseStudyIdExtractor class.

The BaseStudyIdExtractor class is a base class for study ID extraction. It provides a default implementation
for extracting the study ID from a given path.
Each medical imaging examination contains many imgs, study id identifies all the imgs from the same examination.
"""
from .base_id import BaseIdExtractor


class BaseStudyIdExtractor(BaseIdExtractor):
    """
    Base class for study ID extractors.

    This class provides a base implementation for extracting study IDs from file paths.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def _extract(self, img_path: str) -> str:
        """
        Extract the study ID from the given path.

        Args:
            path (str): The path from which to extract the study ID.

        Returns:
            str: The extracted study ID.
        """
        return img_path
