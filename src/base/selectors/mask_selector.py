"""
This module contains the definition of the BaseMaskSelector class.

The BaseMaskSelector class is a base class for Masks selection. It provides a default implementation
for checking whether a file is the intended mask based on the given path.
"""


class BaseMaskSelector:
    """
    Base class for Mask selection.

    This class provides a default implementation for checking whether a file is the intended mask based on the given path.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, mask_prefix: str, path: str) -> bool:
        """
        Check if the file is the intended mask.

        Args:
            mask_prefix (str): The prefix of the mask file.
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended mask, False otherwise.
        """
        return self._is_mask_file(mask_prefix, path)

    def _is_mask_file(self, mask_prefix: str, path: str) -> bool:
        """
        Check if the file is the intended mask.

        Args:
            mask_prefix (str): The prefix of the mask file.
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended mask, False otherwise.
        """
        return mask_prefix in path

    def is_not_mask_file(self, mask_prefix: str, path: str) -> bool:
        """
        Check if the file is not the intended mask.

        Args:
            mask_prefix (str): The prefix of the mask file.
            path (str): The file path to check.

        Returns:
            bool: True if the file is not the intended mask, False otherwise.
        """
        return mask_prefix not in path
