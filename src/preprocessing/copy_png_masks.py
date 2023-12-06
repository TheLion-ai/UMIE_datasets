"""Copy PNG masks to a new folder structure."""
import glob
import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class CopyPNGMasks(TransformerMixin):
    """Copy PNG masks to a new folder structure."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        masks_path: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        mask_selector: str = "segmentations",
        **kwargs: dict,
    ):
        """Copy PNG masks to a new folder structure.

        Args:
            target_path (str): Path to the target folder.
            dataset_name (str): Name of the dataset.
            masks_path (str): Path to source folder with masks
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str, optional): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            mask_selector (str, optional): String to select masks. Defaults to "segmentations".
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.masks_path = masks_path
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.mask_selector = mask_selector

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Copy PNG masks to a new folder structure.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Copying PNG masks...")
        # for img_path in tqdm(X):
        #     self.copy_png_masks(img_path)
        # mask_paths = (
        #     glob.glob(f"{self.masks_path}/*.tif", recursive=True)
        #     + glob.glob(f"{self.masks_path}/*.tiff", recursive=True)
        #     + glob.glob(f"{self.masks_path}/*.png", recursive=True)
        #     + glob.glob(f"{self.masks_path}/*.jpg", recursive=True)
        #     + glob.glob(f"{self.masks_path}/*.jpeg", recursive=True)
        # )
        mask_paths = []
        for root, dirnames, filenames in os.walk(self.masks_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                else:
                    mask_paths.append(os.path.join(root, filename))
        if mask_paths:
            for img_path in tqdm(mask_paths):
                self.copy_png_masks(img_path)
        return X

    def copy_png_masks(self, img_path: str) -> None:
        """Copy PNG masks to a new folder structure.

        Args:
            img_path (str): Path to the image.
        """
        img_id = self.img_id_extractor(img_path)
        if self.mask_selector in img_id:
            img_id = img_id.replace(self.mask_selector, "")
        # if phase_id in self.phases.keys():
        #     return None
        if self.mask_selector not in img_path:
            return None
        else:
            if len(self.phases.keys()) <= 1:
                # new_file_name = f"{self.dataset_uid}_{study_id}_{img_id}"
                new_file_name = img_id
                new_path = os.path.join(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    self.mask_folder_name,
                    new_file_name,
                )
                if not os.path.exists(new_path):
                    shutil.copy2(img_path, new_path)
            else:
                phase_id = self.phase_extractor(img_path)
                for phase_id in self.phases.keys():
                    if phase_id == self.phase_extractor(img_path):
                        phase_name = self.phases[phase_id]
                        # new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
                        new_file_name = img_id
                        new_path = os.path.join(
                            self.target_path,
                            f"{self.dataset_uid}_{self.dataset_name}",
                            phase_name,
                            self.mask_folder_name,
                            new_file_name,
                        )

                        if not os.path.exists(new_path):
                            shutil.copy2(img_path, new_path)
