"""
This module contains the BaseLabelExtractor class.

BaseLabelExtractor is a base class for constructing a label extractor for an individual dataset.
Label extractors are used to extract labels from the labels file.
Each label extractor at initialization should receive a dictionary of mapping source labels to target labels dict (as seen in labels in dataset config).
Label extractors should implement the _extract method to extract the label from the image path or mask_path.
There is no default implementation of the _extract method, so it must be implemented in the derived class.
"""

from abc import ABC, abstractmethod


class BaseLabelExtractor(ABC):
    """Base class for constructing a label extractor for an individual dataset."""

    def __init__(self, labels: dict):
        """Initialize the BaseLabelExtractor class."""
        self.labels = labels

    def __call__(self, img_path: str, mask_path: str) -> list:
        """
        Extract labels from the labels file.

        Args:
            img_path (str): The path to the image file.

        Returns:
            list: A list of labels extracted from the labels file.
        """
        return self._extract(img_path, mask_path)

    @abstractmethod
    def _extract(self, img_path: str, mask_path: str) -> list:
        """
        Extract the label from the image path.

        Args:
            img_path (str): The path of the image file.

        Returns:
            list: The extracted label.
        """
        raise NotImplementedError("The method _extract must be implemented in the derived class.")
