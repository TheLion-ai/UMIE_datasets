"""Change img ids to match the format of the rest of the dataset."""
import glob
import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddNewIds(TransformerMixin):
    """Change img ids to match the format of the rest of the dataset."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        segmentation_dcm_prefix: str = "segmentations",
        **kwargs: dict,
    ):
        """Change img ids to match the format of the rest of the dataset.

        Args:
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            segmentation_dcm_prefix (str, optional): String to select masks. Defaults to "segmentations".
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.segmentation_dcm_prefix = segmentation_dcm_prefix

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
        print("Adding new ids to the dataset...")
        for img_path in tqdm(X):
            self.add_new_ids(img_path)

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
            phase_id = "0"
            new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.image_folder_name,
                new_file_name,
            )
        else:
            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            elif self.segmentation_dcm_prefix in img_path:
                return None
            phase_name = self.phases[phase_id]
            new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.image_folder_name,
                new_file_name,
            )

        if not os.path.exists(new_path):  # for img_path in tqdm(X):
            #     if img_path.endswith(".nii.gz"):
            #         self.convert_nii2dcm(img_path)
            shutil.copy2(img_path, new_path)
