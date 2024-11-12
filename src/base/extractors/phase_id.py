"""
This module contains the definition of the BasePhaseIdExtractor class.

The BasePhaseIdExtractor class is a base class for phase ID extractors.
It provides a default implementation for extracting the phase ID from a given path.
Subclasses must override the `_extract` method to provide custom extraction logic.
Phases are the different stages of the medical imaging examination process, e.g. pre-contrast CT, arterial phase CT, venous phase CT, etc.
For most datasets we use broad categories like CT, MRI, X-ray.
"""
from abc import ABC

from .base_id import BaseIdExtractor


class BasePhaseIdExtractor(BaseIdExtractor, ABC):
    """
    Base abstract class for phase ID extractors.

    Subclasses must override the `_extract` method inherited from BaseExtractor to provide custom extraction logic.

    Attributes:
        phases: dict[int, str] Dictionary containing mapping of phases to phase IDs

    Methods:
        __call__(self, path: str) -> str: Invokes the phase ID extraction process.
        _extract(self, path: str) -> str: Default implementation of the phase ID extraction.

    """

    phases: dict[int, str]

    def __init__(self, phases: dict[int, str]):
        """
        Initialize the BasePhaseIdExtractor class.

        Args:
            phases (dict[int, str]): A dictionary mapping phase IDs to their corresponding names.

        Returns:
            None
        """
        self.phases = phases

    def _get_phase_id_from_dict(self, phase_name: str) -> str:
        for key, value in self.phases.items():
            if phase_name.casefold() == value.casefold():
                return str(key)
        print(f"Phase name: {phase_name} not found!")
        return ""


class DefaultPhaseIDExtractor(BasePhaseIdExtractor):
    """
    This class provides a default implementation for extracting the phase ID from a given path.

    Attributes:
        None

    Methods:
        _extract(self, path: str) -> str: Default implementation of the phase ID extraction.

    """

    def _extract(self, img_path: str) -> str:
        """
        By default, each dataset has a single phase, so we do not need any extraction logic.

        Args:
            path (str): The path from which to extract the phase ID.

        Returns:
            str: Returns 0 as a string.
        """
        return self._return_zero()
