"""
This module contains the definition of the BaseModalityIdExtractor class.

The BaseModalityIdExtractor class is a base class for modality ID extractors.
It provides a default implementation for extracting the modality ID from a given path.
Subclasses can override the `_extract` method to provide custom extraction logic.
The modality is the broad imaging type of the examination, e.g. CT, MRI, X-ray, MG, PET.
(A finer-grained contrast *phase* - arterial / venous / delayed - is a separate, future concept;
see ``Modalities Ontology.md`` and the Task 47 decision note. It is intentionally not modelled here.)
"""

from .base_id import BaseIdExtractor


class BaseModalityIdExtractor(BaseIdExtractor):
    """
    Base class for modality ID extractors.

    This class provides a default implementation for extracting the modality ID from a given path.
    Subclasses can override the `_extract` method to provide custom extraction logic.

    Attributes:
        None

    Methods:
        __call__(self, path: str) -> str: Invokes the modality ID extraction process.
        _extract(self, path: str) -> str: Default implementation of the modality ID extraction.

    """

    modalities: dict[int, str]

    def __init__(self, modalities: dict[int, str]):
        """
        Initialize the BaseModalityIdExtractor class.

        Args:
            modalities (dict[int, str]): A dictionary mapping modality IDs to their corresponding names.

        Returns:
            None
        """
        self.modalities = modalities

    def _extract(self, path: str) -> str:
        """
        Use default implementation of the modality ID extraction.

        This method extracts the base name of the path as the modality ID.

        Args:
            path (str): The path from which to extract the modality ID.

        Returns:
            str: The extracted modality ID.
        """
        # By default, each dataset has a single modality, so we do not need any extraction logic.
        return "0"

    def _get_modality_id_from_dict(self, modality_name: str) -> str:
        """
        Get modality ID from modalities dict based on the modality name.

        Args:
            modality_name (str): Name of the modality

        Returns:
            str: Modality ID. If none found it returns an empty string.
        """
        for key, value in self.modalities.items():
            if modality_name.casefold() == value.casefold():
                return str(key)
        print(f"Modality name: {modality_name} not found!")
        return ""
