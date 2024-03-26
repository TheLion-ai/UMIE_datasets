"""Copy PNG masks to a new folder structure."""

import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class CopyMasks(TransformerMixin):
    """Copy PNG masks to a new folder structure."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str,
        mask_folder_name: str,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        img_prefix: str = "imaging",
        segmentation_prefix: str = "segmentation",
        mask_selector: str = "segmentations",
        **kwargs: dict,
    ):
        """Copy PNG masks to a new folder structure.

        Args:
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            segmentation_prefix (str, optional): String to select masks. Defaults to "segmentations".
            mask_selector (str, optional): String to distinguish between images and masks. Defaults to "segmentations".
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.img_prefix = img_prefix
        self.segmentation_prefix = segmentation_prefix
        self.mask_selector = mask_selector

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Copy masks to a new folder structure.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Copying masks...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            path_el = img_path.rsplit(self.img_prefix, 1)
            mask_path = self.segmentation_prefix.join(path_el)
            if os.path.exists(mask_path):
                self.copy_masks(mask_path)
        return X

    def copy_masks(self, img_path: str) -> None:
        """Copy PNG masks to a new folder structure.

        Args:
            img_path (str): Path to the image.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        # TODO: remove duplicate code from add_new_ids.py, Move this step to add_new_ids???
        if self.mask_selector in img_id:
            img_id = img_id.replace(self.mask_selector, "")
        if self.segmentation_prefix not in img_path:
            return None
        for phase_id in self.phases.keys():
            if phase_id == self.phase_extractor(img_path):
                phase_name = self.phases[phase_id]
                new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
                if "." not in new_file_name:
                    new_file_name = new_file_name + ".png"
                new_path = os.path.join(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase_name,
                    self.mask_folder_name,
                    new_file_name,
                )

                if not os.path.exists(new_path):
                    print("copied mask: ", new_path)
                    shutil.copy2(img_path, new_path)
