"""
This module contains the definition of the BaseXmlMaskCreator class.

BaseXmlMaskCreator is an abstract base class for creating image masks from xml file paths.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Callable


class BaseXmlMaskCreator(ABC):
    """Abstract base class for extracting image IDs from file paths."""

    def __call__(self, path: str) -> None:
        """Create mask from xml file.

        Args:
            path (str): The file path from which to get xml file data for mask creation.

        Returns:
            None
        """
        # Get instance of caller class (some step class), to be able
        # to use BaseStep functions and variables
        current_frame = inspect.currentframe()
        if current_frame is not None:
            previous_frame = current_frame.f_back
            if previous_frame is not None:
                caller = previous_frame.f_locals["self"]
                return self._create(path, caller)

        raise ValueError("current_frame or previous_frame is None")

    @abstractmethod
    def _create(self, path: str, caller: Callable) -> None:
        """Abstract placeholder method.

           Method with this name should be implemented in pipeline script file.

        Args:
            path (str): The file path from which to get xml file data for mask creation.
            caller (Callable): Caller step class instance to be able to access
                               BaseStep functions and variables
        """
        raise NotImplementedError("_create method not implmented for XmlMaskCreator.")
