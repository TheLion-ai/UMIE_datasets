"""Add labels to the images and masks based on the labels.json file."""
import glob
import json
import os
import shutil
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm

from ..utils.get_images_target_paths import get_images_paths


class AddLabels(TransformerMixin):
    """Add labels to the images and masks based on the provided get_label function or labels.json file."""

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
        zfill: int = 3,
        labels_path: str = "",
        get_label: Callable = lambda x: [],
        **kwargs: dict,
    ) -> None:
        """Add labels to the images and masks based on the provided get_label function or labels.json file.

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
        labels_list = []
        if self.labels_path:
            labels_path_extention = os.path.basename(self.labels_path).split(".")[1]
            if labels_path_extention == "json":
                with open(self.labels_path) as f:
                    labels_list = json.load(f)

        print("Adding labels...")
        for img_path in tqdm(X):
            self.add_labels(img_path, labels_list)
        new_paths = get_images_paths(
            self.phases,
            self.target_path,
            self.dataset_uid,
            self.dataset_name,
            self.image_folder_name,
        )
        return new_paths

    def add_labels(self, img_path: str, labels_list: list) -> None:
        """Add labels to the image and mask.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        img_id_full = self.img_id_extractor(img_path)
        img_id = img_id_full.split(".")[0]
        label_prefix = "-"
        if labels_list != []:
            labels = "".join([label_prefix + label for label in labels_list])
        else:
            labels = self.get_label(img_path)
        labels_str = "".join([label_prefix + label for label in labels])
        new_name = f"{img_id}{labels_str}.png"
        if len(self.phases.keys()) <= 1:
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                self.image_folder_name,
                new_name,
            )
        else:
            phase_id = self.phase_extractor(img_path)
            if phase_id not in self.phases.keys():
                return None
            phase_name = self.phases[phase_id]
            new_path = os.path.join(
                self.target_path,
                f"{self.dataset_uid}_{self.dataset_name}",
                phase_name,
                self.image_folder_name,
                new_name,
            )

        if not os.path.exists(new_path):
            shutil.copy2(img_path, new_path)

        root_path = os.path.dirname(os.path.dirname(new_path))
        mask_path = os.path.join(root_path, self.mask_folder_name, img_id_full)
        if os.path.exists(mask_path):
            os.rename(
                mask_path,
                f"{root_path}/{self.mask_folder_name}/{img_id}{labels_str}.png",
            )
