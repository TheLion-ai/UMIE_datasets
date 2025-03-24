"""Validate the dataset output data."""
import os

import cv2
import jsonlines
from tqdm import tqdm

from base.step import BaseStep
from config import labels


class ValidateData(BaseStep):
    """Validate the dataset output data."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Validate the dataset output data.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Validating output data...")
        self.validate_dataset()
        print("Validation complete.")
        return X

    def validate_dataset(
        self,
    ) -> None:
        """Validate the dataset output data."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Dataset jsonl file {self.json_path} does not exist")
        print("Validating output data...")

        self._validate_jsonl(self.json_path)

    def _validate_jsonl(self, json_path: str) -> None:
        with jsonlines.open(json_path, mode="r") as reader:
            for obj in tqdm(reader):
                try:
                    self._validate_object(obj)
                except Exception as e:
                    print(f"Error in image {obj['umie_path']}: {e}")

    def _validate_object(self, obj: dict) -> None:
        self._validate_string_keys(obj)
        self._validate_image(obj)
        self._validate_mask(obj)
        self._validate_labels(obj)

    def _validate_string_keys(self, obj: dict) -> None:
        str_keys = ["umie_path", "dataset_name", "dataset_uid", "phase_name", "study_id", "umie_id"]
        for str_key in str_keys:
            if str_key not in obj.keys():
                print(f"Key {str_key} not in {obj['umie_path']}")
                continue
            if not isinstance(obj[str_key], str) or len(obj[str_key]) == 0:
                print(f"{str_key} is not a string or is empty in {obj['umie_path']}")

    def _validate_image(self, obj: dict) -> None:
        umie_path = os.path.join(self.target_path, obj["umie_path"])
        img = cv2.imread(umie_path)
        if img is None:
            print(f"Image {obj['umie_path']} is not found.")

    def _validate_mask(self, obj: dict) -> None:
        if not isinstance(obj["mask_path"], str):
            print("mask_path is not a string.")
        elif len(obj["mask_path"]) != 0:
            mask_path = os.path.join(self.target_path, obj["mask_path"])
            mask = cv2.imread(mask_path)
            if mask is None:
                print(f"Mask {obj['mask_path']} is not found.")

    def _validate_labels(self, obj: dict) -> None:
        if not isinstance(obj["labels"], list):
            print("labels is not a list.")
        else:
            for label in obj["labels"]:
                if not isinstance(label, dict):
                    print("label is not a dictionary.")
                else:
                    label_key = list(label.keys())[0]
                    label_value = list(label.values())[0]
                    if not isinstance(label_key, str) or len(label_key) == 0:
                        print("label key is not a string or is empty.")
                    elif not isinstance(label_value, (int, float)):
                        print("label grade is not a number")
                    elif label_key not in dir(labels):
                        print(f"Label {label_key} is not in config/labels.py.")
