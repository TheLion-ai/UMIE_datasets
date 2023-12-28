"""Base class for all steps in the pipeline."""
import os
from abc import abstractmethod

from sklearn.base import TransformerMixin


class BaseStep(TransformerMixin):
    """Base class for all steps in the pipeline."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str,
        mask_folder_name: str,
        **kwargs: dict,
    ):
        """Create class for all steps in the pipeline.

        Args:
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str): Name of the folder with images.
            mask_folder_name (str): Name of the folder with masks.
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name

    @abstractmethod
    def transform(X) -> list:
        """Transform data.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        return X

    def create_file_path(self, slice_id: str, study_id: str, phase_id: str, mask: bool) -> str:
        """Create path for the image or mask.

        Args:
            slice_id (str): Slice id.
            study_id (str): Study id.
            phase_id (str): Phase id.
            mask (bool): Whether to create path for the mask or image.
        Returns:
            new_path (str): Path to the image or mask.
        """
        new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{slice_id}.png"
        phase_name = self.phases[phase_id]
        name = self.mask_folder_name if mask else self.image_folder_name
        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            name,
            new_file_name,
        )
        return new_path
