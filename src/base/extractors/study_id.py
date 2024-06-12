import os

class BaseStudyIdExtractor():
    def __call__(self, path: str) -> str:

        return self._extract(path)

    def _extract(self, path: str) -> str:
        """Default implementation of the study id extraction. It extracts the base name of the path.

        Args:
            path (str): The path from which to extract the base name.

        Returns:
            str: The extracted base name.
        """
        return path