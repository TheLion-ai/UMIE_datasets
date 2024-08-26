"""Add labels to the images and masks based on the labels.json file. The step requires the pipeline to specify the function for mapping the images with annotations."""
import glob
import json
import os
from typing import Callable, Optional

import jsonlines
from tqdm import tqdm

from umie_datasets.base.extractors.img_id import BaseImgIdExtractor
from umie_datasets.base.step import BaseStep


class AddLabels(BaseStep):
    """Add labels to the images and masks based on the function for mapping the images with annotations specified by the pipeline."""

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
        if os.path.exists(os.path.join(self.target_path, "source_paths.json")):
            source_paths_dict = json.load(open(os.path.join(self.target_path, "source_paths.json")))
        else:
            source_paths_dict = None

        self.json_updates: dict = {}
        for img_path in tqdm(X):
            self.add_labels(img_path, source_paths_dict)

        updated_lines = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                if obj["umie_path"] in self.json_updates.keys():
                    obj["labels"] = self.json_updates[obj["umie_path"]]
                updated_lines.append(obj)

        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in updated_lines:
                writer.write(obj)

        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        new_paths = glob.glob(os.path.join(root_path, f"**/{self.image_folder_name}/*.png"), recursive=True)
        return new_paths

    def add_labels(self, img_path: str, source_path_dict: Optional[dict] = None) -> None:
        """Add labels to the image and mask based on the label_extractor function specified by the pipeline.

        Args:
            img_path (str): Path to the image.
            labels_list (list): List of labels.
        """
        mask_path = self.get_umie_mask_path(img_path)
        if source_path_dict:
            labels = self.label_extractor(source_path_dict[img_path], mask_path)
        else:
            labels = self.label_extractor(img_path, mask_path)
        if labels:
            key = self.get_path_without_target_path(img_path)
            self.json_updates[key] = labels
