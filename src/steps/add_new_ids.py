"""Change img ids to match the format of the rest of the dataset."""
import glob
import os
import shutil
from typing import Callable

from tqdm import tqdm

from src.steps.base_step import BaseStep


class AddNewIds(BaseStep):
    """Change img ids to match the format of the rest of the dataset."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        segmentation_dcm_prefix: str = "segmentations",
        masks_path: str = "",
        use_siuid_as_index: bool = False,
        zfill: int = 4,
        **kwargs: dict,
    ):
        """Change img ids to match the format of the rest of the dataset.

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
            segmentation_dcm_prefix (str, optional): String to select masks. Defaults to "segmentations".
            masks_path (str, optional): Path to the folder with masks. Defaults to "".
            use_siuid_as_index (bool, optional): Whether to use SOPInstanceUID as image id. Defaults to False.
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 4.
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
        self.segmentation_dcm_prefix = segmentation_dcm_prefix
        self.masks_path = masks_path
        self.use_siuid_as_index = use_siuid_as_index
        self.zfill = zfill

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
        for idx, img_path in tqdm(enumerate(X)):
            self.add_new_ids(img_path, idx)

        # TODO: Fix for multiple phase
        root_path = os.path.join(
            self.target_path, f"{self.dataset_uid}_{self.dataset_name}", "CT", self.image_folder_name
        )
        new_paths = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)
        return new_paths

    def add_new_ids(self, img_path: str, idx: int) -> None:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            img_path (str): Path to the image.
        """
        # Extract relevant information from the source path
        # The logic of the extraction functions depends on the dataset
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_extractor(img_path)
        if phase_id not in self.phases.keys():
            raise ValueError(f"Phase {phase_id} not in the phases dictionary.")

        # phase_name = self.phases[phase_id]
        # new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
        # new_path = os.path.join(
        #     self.target_path,
        #     f"{self.dataset_uid}_{self.dataset_name}",
        #     phase_name,
        #     self.image_folder_name,
        #     new_file_name,
        # )
        if self.use_siuid_as_index:
            img_id = img_id.split("_")[-1]
            img_id = idx
            new_img_path = self.create_file_path(img_id, study_id, phase_id, mask=False)
            shutil.copy2(img_path, new_img_path)
            mask_path = img_path.replace(self.image_folder_name, self.mask_folder_name)
            if os.path.exists(mask_path):
                new_mask_path = self.create_file_path(img_id, study_id, phase_id, mask=True)
                shutil.copy2(mask_path, new_mask_path)
            return

        if self.segmentation_dcm_prefix in img_path:
            new_path = self.create_file_path(img_id, study_id, phase_id, mask=True)
        else:
            new_path = self.create_file_path(img_id, study_id, phase_id, mask=False)

        if not os.path.exists(new_path):
            shutil.copy2(img_path, new_path)
