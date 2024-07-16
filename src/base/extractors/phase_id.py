"""
This module contains the definition of the BasePhaseIdExtractor class.

The BasePhaseIdExtractor class is a base class for phase ID extractors.
It provides a default implementation for extracting the phase ID from a given path.
Subclasses can override the `_extract` method to provide custom extraction logic.
Phases are the different stages of the medical imaging examination process, e.g. pre-contrast CT, arterial phase CT, venous phase CT, etc.
For most datasets we use broad categories like CT, MRI, X-ray.
"""


class BasePhaseIdExtractor:
    """
    Base class for phase ID extractors.

    This class provides a default implementation for extracting the phase ID from a given path.
    Subclasses can override the `_extract` method to provide custom extraction logic.

    Attributes:
        None

    Methods:
        __call__(self, path: str) -> str: Invokes the phase ID extraction process.
        _extract(self, path: str) -> str: Default implementation of the phase ID extraction.

    """

    def __init__(self, phases: dict[int, str]):
        """
        Initialize the BasePhaseIdExtractor class.

        Args:
            phases (dict[int, str]): A dictionary mapping phase IDs to their corresponding names.

        Returns:
            None
        """
        self.phases = phases

    def __call__(self, path: str) -> str:
        """
        Invoke the phase ID extraction process.

        Args:
            path (str): The path from which to extract the phase ID.

        Returns:
            str: The extracted phase ID.
        """
        return self._extract(path)

    def _extract(self, path: str) -> str:
        """
        Use default implementation of the phase ID extraction.

        This method extracts the base name of the path as the phase ID.

        Args:
            path (str): The path from which to extract the phase ID.

        Returns:
            str: The extracted phase ID.
        """
        # By default, each dataset has a single phase, so we do not need any extraction logic.
        return "0"
