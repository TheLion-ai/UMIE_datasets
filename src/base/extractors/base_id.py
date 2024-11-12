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
        raise NotImplementedError("The method _extract must be implemented in the derived class.")

    @staticmethod
    def _extract_filename(input_path: str, with_suffix: bool = False) -> str:
        """
        Extract the filename without its suffix from a given file path.

        Args:
            input_path (str): The path to the file.

        Returns:
            str: The filename, without its suffix (extension) and directory path.
        """
        if with_suffix:
            return Path(input_path).name
        else:
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

    def _extract_by_separator(self, path: str, separator: str, index: int = -1, maxsplit: int = -1) -> str:
        """Extract image ID from the path by the separator.

        Args:
            path (str): The file path from which to extract the image ID.
            separator (str): The separator to split the path.
            index (int): The index of the image ID in the split path. Defaults to -1.
            maxsplit (int): The maximum number of splits to perform. Defaults to -1.

        Returns:
            str: The extracted image ID.
        """
        return Path(path).name.rsplit(separator, maxsplit=maxsplit)[index]

    @staticmethod
    def _return_zero(suffix: str = "") -> str:
        """Return "0" with appropriate suffix.

        Returns:
            str: "0" + suffix.
        """
        return "0" + suffix
