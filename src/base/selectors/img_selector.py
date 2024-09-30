"""
This module contains the definition of the BaseImageSelector class.

The BaseImageSelector class is a base class for Images selection. It provides a default implementation
for checking whether a file is the intended image based on the given path.
"""


class BaseImageSelector:
    """
    Base class for Image selection.

    This class provides a default implementation for checking whether a file is the intended image based on the given path.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, path: str) -> bool:
        """
        Check if the file is the intended image.

        Args:
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended image, False otherwise.

        """
        return self._is_image_file(path)

    def _is_image_file(self, path: str) -> bool:
        """
        Check if the file is the intended image.

        Args:
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended image, False otherwise.
        """
        return True
