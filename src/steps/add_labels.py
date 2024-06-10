"""Add labels to the images and masks based on the labels.json file. The step requires the pipeline to specify the function for mapping the images with annotations."""
import csv
import glob
import os
from typing import Callable

import jsonlines
from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddLabels(TransformerMixin):
    """Add labels to the images and masks based on the function for mapping the images with annotations specified by the pipeline."""

    def __init__(
        self,
        target_path: str,
        json_path: str,
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
            json_path: (str): path to jsonlines with info about individual image in the target dataset,
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            window_width (int): Window width for the images.
            image_folder_name (str): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 3.
            labels_path (str, optional): Path to the labels file. Defaults to "".
            get_label (Callable, optional): Function to get the label. Defaults to lambda x: [].
        """
        self.target_path = target_path
        self.json_path = json_path
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
        self.paths_data: dict[str, str] = {}

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
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        if os.path.exists(os.path.join(self.target_path, "source_paths.csv")):
            self.paths_data = dict(list(csv.reader(open(os.path.join(self.target_path, "source_paths.csv")))))

        self.json_updates: dict = {}
        for img_path in tqdm(X):
            self.add_labels(img_path)

        updated_lines = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                if obj["file_name"] in self.json_updates.keys():
                    obj["labels"] = self.json_updates[obj["file_name"]]
                updated_lines.append(obj)

        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in updated_lines:
                writer.write(obj)

        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        new_paths = glob.glob(os.path.join(root_path, f"**/{self.image_folder_name}/*.png"), recursive=True)
        return new_paths

    def add_labels(self, img_path: str) -> None:
        """Add labels to the image and mask based on the get_label function specified by the pipeline.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        img_id = os.path.basename(img_path).split(".")[0]
        if os.path.exists(os.path.join(self.target_path, "source_paths.csv")):
            labels = self.get_label(self.paths_data[img_id])
        else:
            labels = self.get_label(img_path)
        if labels:
            self.json_updates[img_id] = labels
