"""
This module contains the definition of the BaseIdExtractor class.

The BaseIdExtractor class is an abstract base class for extraction of information from image path.
It provides a default implementation as well as helper methods for its children.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseIdExtractor(ABC):
    """
    Base class for ID extractors.

    This class provides a base implementation for extracting IDs from file paths.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, path: str) -> str:
        """
        Call the _extract method to extract desired information from the given path.

        Args:
            path (str): The path from which to extract the desired information.

        Returns:
            str: The extracted information.
        """
        return self._extract(path)

    @abstractmethod
    def _extract(self, img_path: str) -> str:
        """
        Extract information from the given path.

        Args:
            path (str): The path from which to extract the desired information.

        Returns:
            str: The extracted information.
        """
        raise NotImplementedError

    @staticmethod
    def _extract_filename(input_path: str) -> str:
        """
        Extract the filename without its suffix from a given file path.

        Args:
            input_path (str): The path to the file.

        Returns:
            str: The filename, without its suffix (extension) and directory path.
        """
        return Path(input_path).with_suffix("").stem

    @staticmethod
    def _extract_parent_dir(
        img_path: str,
        parent_dir_level: int = 1,
        include_path: bool = True,
    ) -> str:
        """
        Extract the parent directory of a given file path, with options for depth and base name only.

        Args:
            img_path (str): The path to the file.
            parent_dir_level (int, optional): The number of levels up to go from the file path.
                                Defaults to 1, which gets the immediate parent directory.
            include_path (bool, optional): If False, returns only the directory name (stem) of the whole directory path.
                                            Defaults to False, returning the full directory path.

        Returns:
            str: The parent directory path (or its base name if `include_path` is False) at the specified level.
        """
        output_path = Path(img_path)

        for _ in range(abs(parent_dir_level)):
            output_path = output_path.parent

        if not include_path:
            return output_path.stem

        return str(output_path)
