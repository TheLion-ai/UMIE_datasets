"""Change masks ids to match the format of the rest of the dataset."""
import glob
import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddNewIdsMasks(TransformerMixin):
    """Change masks ids to match the format of the rest of the dataset."""

    def __init__(
        self,
        target_path: str,
        masks_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        mask_selector: str = "segmentations",
        mask_folder_name: str = 'Masks',
        **kwargs: dict,
    ):
        """Change masks ids to match the format of the rest of the dataset.

        Args:
            target_path (str): Path to the target folder.
            masks_path (str): Path to the folder with masks.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            mask_selector (str, optional): String to select masks. Defaults to "segmentations".
            mask_folder_name (str): Name of the folder with masks. Defaults to "Masks".
        """
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.mask_selector = mask_selector
        self.mask_folder_name = mask_folder_name

    def transform(
        self,
        X: list,
    ) -> list:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Adding new ids to masks...")
        mask_paths = glob.glob(f"{self.masks_path}/*.png", recursive=True)
        if mask_paths:
            for img_path in tqdm(mask_paths):
                self.add_new_ids(img_path)
        else:
            print("Masks not found.")

        root_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            self.image_folder_name,
        )
        new_paths = glob.glob(f"{root_path}/*.*", recursive=True)
        return new_paths

    def add_new_ids(self, img_path: str) -> None:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            img_path (str): Path to the image.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)

        if len(self.phases.keys()) <= 1:
            new_file_name = f"{self.dataset_uid}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.mask_folder_name,
                new_file_name,
            )
        else:
            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            elif self.mask_selector in img_path:
                return None
            phase_name = self.phases[phase_id]
            new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.mask_folder_name,
                new_file_name,
            )

        if not os.path.exists(new_path):
            shutil.copy2(img_path, new_path)
