from abc import ABC, abstractmethod

class BaseLabelExtractor(ABC):
    """Base class for constructing a label extractor for an individual dataset. Label"""
    def __init__(self, labels: dict):
        """
        Initialize the BaseLabelExtractor class.

        Args:
            labels (dict): A dictionary containing labels for the dataset.
        """
        self.labels = labels
    
    def __call__(self, img_path: str) -> list:
        """
        Extract labels from the labels file.

        Args:
            img_path (str): The path to the image file.

        Returns:
            list: A list of labels extracted from the labels file.
        """
        return self._extract(img_path)
    
    @abstractmethod
    def _extract(self, img_path: str) -> list:
        """
        Extract the label from the image path.

        Args:
            img_path (str): The path of the image file.

        Returns:
            list: The extracted label.
        """
        raise NotImplementedError("The method _extract must be implemented in the derived class.")
