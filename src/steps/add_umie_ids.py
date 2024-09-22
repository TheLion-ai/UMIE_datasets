"""Change img ids to match the format of the rest of the dataset."""
import glob
import json
import os
import shutil
from typing import Callable, Optional

import jsonlines
import numpy as np
from tqdm import tqdm

from base.extractors.img_id import BaseImgIdExtractor
from base.step import BaseStep


class AddUmieIds(BaseStep):
    """Change img ids to match the format of the rest of the dataset."""

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
        if len(X) == 0:
            raise ValueError("No list of files provided.")

        self.new_json: list = []
        X.sort(key=lambda x: self.get_umie_id(x))  # Sort the list of paths based on umie_id
        for img_path in tqdm(X):
            self.add_umie_ids(img_path)

        # Update JSON file
        with jsonlines.open(self.json_path, "w") as writer:
            for obj in self.new_json:
                writer.write(obj)

        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        new_paths = glob.glob(os.path.join(root_path, f"**/{self.image_folder_name}/*.png"), recursive=True)
        return new_paths

    def _update_json(self, umie_path: str, mask_path: str) -> None:

        phase_name, study_id, img_id = self.decode_umie_img_path(umie_path)

        """Update JSON file with the infomration about the images."""
        comparative = ""
        if "PRE" in img_id:
            comparative = "PRE"
        elif "POST" in img_id:
            comparative = "POST"

        if mask_path != "":
            mask_path = self.get_path_without_target_path(mask_path)

        img_info = {}
        img_info = {
            "umie_path": self.get_path_without_target_path(umie_path),
            "dataset_name": self.dataset_name,
            "dataset_uid": self.dataset_uid,
            "phase_name": phase_name,
            "comparative": comparative,
            "study_id": str(study_id),
            "umie_id": os.path.basename(umie_path),
            "mask_path": mask_path,
            "labels": [],
        }

        self.new_json.append(img_info)

    def add_umie_ids(self, img_path: str) -> None:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            img_path (str): Path to the image.
        """
        # Extract relevant information from the source path
        # The logic of the extraction functions depends on the dataset

        if self.segmentation_prefix is not None and self.segmentation_prefix in img_path:
            return None

        umie_path = self.get_umie_img_path(img_path)

        if not self.validate_umie_path(img_path):
            return None

        if not os.path.exists(umie_path):
            if self.target_path in img_path:
                os.rename(img_path, umie_path)
            else:
                shutil.copy2(img_path, umie_path)

        umie_mask_path = ""
        if self.mask_folder_name is not None:
            old_mask_path = img_path.replace(self.image_folder_name, self.mask_folder_name)
            new_mask_path = umie_path.replace(self.image_folder_name, self.mask_folder_name)
            if os.path.exists(new_mask_path):
                umie_mask_path = new_mask_path
            if self.mask_folder_name in img_path:
                if os.path.exists(old_mask_path):
                    os.rename(old_mask_path, new_mask_path)
                    umie_mask_path = new_mask_path

        self._update_json(umie_path, umie_mask_path)
