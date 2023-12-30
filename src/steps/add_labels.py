"""Add labels to the images and masks based on the labels.json file. The step requires the pipeline to specify the function for mapping the images with annotations."""
import glob
import json
import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddLabels(TransformerMixin):
    """Add labels to the images and masks based on the function for mapping the images with annotations specified by the pipeline."""

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
        zfill: int = 3,
        labels_path: str = "",
        get_label: Callable = lambda x: [],
        **kwargs: dict,
    ):
        """Add labels to the images and masks based on the labels.json file.

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
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 3.
            labels_path (str, optional): Path to the labels file. Defaults to "".
            get_label (Callable, optional): Function to get the label. Defaults to lambda x: [].
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
        self.zfill = zfill
        self.labels_path = labels_path
        self.get_label = get_label

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Add labels to the images and masks.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Adding labels...")
        for img_path in tqdm(X):
            self.add_labels(img_path)
        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        new_paths = glob.glob(os.path.join(root_path, f"**/{self.image_folder_name}/*.png"), recursive=True)
        return new_paths

    def add_labels(self, img_path: str) -> None:
        """Add labels to the image and mask based on the get_label function specified by the pipeline.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        img_id_full = self.img_id_extractor(img_path)
        img_id = img_id_full.split(".")[0]
        label_prefix = "-"  # each label in the target file name is prefixed with this character
        labels = self.get_label(img_path)
        if labels:
            # Add labels to the image path
            labels_str = "".join([label_prefix + label for label in labels])
            new_file_name = f"{img_id}_{labels_str}.png"

            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            phase_name = self.phases[phase_id]
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.image_folder_name,
                new_file_name,
            )

            if not os.path.exists(new_path):
                if self.target_path in img_path:
                    os.rename(img_path, new_path)
                else:
                    shutil.copy2(img_path, new_path)

            # Add labels to the mask path
            mask_path = img_path.replace(self.image_folder_name, self.mask_folder_name)
            if os.path.exists(mask_path):
                os.rename(mask_path, os.path.join(os.path.dirname(mask_path), new_file_name))
