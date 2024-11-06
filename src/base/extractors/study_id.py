"""
This module contains the definition of the BaseStudyIdExtractor class.

The BaseStudyIdExtractor class is a base class for study ID extraction. It provides a default implementation
for extracting the study ID from a given path.
Each medical imaging examination contains many imgs, study id identifies all the imgs from the same examination.
"""
from pathlib import Path


class BaseStudyIdExtractor:
    """
    Base class for study ID extractors.

    This class provides a base implementation for extracting study IDs from file paths.
    Subclasses should override the `_extract` method to provide custom extraction logic.
    """

    def __call__(self, path: str) -> str:
        """
        Call the _extract method to extract the study id from the given path.

        Args:
            path (str): The path from which to extract the study id.

        Returns:
            str: The extracted study id.
        """
        return self._extract(path)

    def _extract(self, img_path: str) -> str:
        """
        Extract the study ID from the given path.

        Args:
            path (str): The path from which to extract the study ID.

        Returns:
            str: The extracted study ID.
        """
        return img_path

    @staticmethod
    def _extract_filename(input_path: str) -> str:
        """
        Extract the filename without its suffix from a given file path.

        Args:
            input_path (str): The path to the file.

        Returns:
            str: The filename, without its extension and directory path.
        """
        return Path(input_path).with_suffix("").stem

    @staticmethod
    def _extract_parent_dir(
        img_path: str,
        node: int = 1,
        basename_only: bool = False,
    ) -> str:
        """
        Extract the parent directory of a given file path, with options for depth and base name only.

        Args:
            img_path (str): The path to the file.
            node (int, optional): The number of levels up to go from the file path.
                                Defaults to 1, which gets the immediate parent directory.
            basename_only (bool, optional): If True, returns only the base name (stem) of the final directory.
                                            Defaults to False, returning the full directory path.

        Returns:
            str: The parent directory path (or its base name if `basename_only` is True) at the specified level.
        """
        output_path = Path(img_path)

        for _ in range(abs(node)):
            output_path = output_path.parent

        if basename_only:
            return output_path.stem

        return str(output_path)
