"""
This module contains the definition of the BaseMaskSelector class.

The BaseMaskSelector class is a base class for Masks selection. It provides a default implementation
for checking whether a file is the intended mask based on the given path.
"""
from abc import ABC, abstractmethod


class BaseMaskSelector(ABC):
    """
    Base class for Mask selection.

    This class provides a default implementation for checking whether a file is the intended mask based on the given path.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, path: str) -> bool:
        """
        Check if the file is the intended mask.

        Args:
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended mask, False otherwise.
        """
        return self._is_mask_file(path)

    @abstractmethod
    def _is_mask_file(self, path: str) -> bool:
        """
        Check if the file is the intended mask.

        Args:
            path (str): The file path to check.

        Returns:
            bool: True if the file is the intended mask, False otherwise.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")
