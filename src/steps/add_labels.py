"""Add labels to the images and masks based on the labels.json file."""
import glob
import json
import os
from typing import Callable

from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddLabels(TransformerMixin):
    """Add labels to the images and masks based on the labels.json file."""

    def __init__(
        self,
        target_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        window_center: int,
        window_width: int,
        image_folder_name: str = "Images",
        mask_folder_name: str = "Masks",
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
            window_center (int): Window center for the images.
            window_width (int): Window width for the images.
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
        self.window_center = window_center
        self.window_width = window_width
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
        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
        return new_paths

    def add_labels(self, img_path: str) -> None:
        """Add labels to the image and mask.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        img_root_path = os.path.dirname(img_path)
        img_id = os.path.basename(img_path).split(".")[0]
        label_prefix = "-"
        labels = self.get_label(img_path)
        if labels:
            labels_str = "".join([label_prefix + label for label in labels])
            new_name = f"{img_id}{labels_str}.png"
            os.rename(img_path, os.path.join(img_root_path, new_name))

            root_path = os.path.dirname(os.path.dirname(img_path))
            mask_path = os.path.join(root_path, self.mask_folder_name, f"{img_id}.png")
            if os.path.exists(mask_path):
                os.rename(mask_path, os.path.join(root_path, self.mask_folder_name, new_name))
