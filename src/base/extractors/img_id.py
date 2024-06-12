from abc import ABC, abstractmethod
import os

class BaseImgIdExtractor(ABC):
    def __call__(self, path: str) -> str:
        """Extract image id from the path.
        e.g. from /path/to/image_001.jpg -> image_001.jpg"""
        return self._extract(path)
    
    def _extract(self, path: str) -> str:
        return os.path.basename(path)