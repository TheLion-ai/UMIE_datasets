"""Preprocessing pipeline for Finding_and_Measuring_Lungs_in_CT_Data dataset."""
import glob
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths


@dataclass
class AlzheimersPipeline(BasePipeline):
    """Preprocessing pipeline for Alzheimers dataset."""

    name: str = field(default="Alzheimers_Dataset")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("convert_jpg2png", ConvertJpg2Png),
            ("add_new_ids", AddNewIds),
            ("add_new_ids", AddLabels),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            # Study id is the folder name of all images in the study
            study_id_extractor=lambda x: "",
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name="Masks",
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return os.path.basename(img_path)

    # Changing labels from dataset (folders names) to match standard
    labels_dict = {
        "MildDemented": "MildDemented",
        "ModerateDemented": "ModerateDemented",
        "NonDemented": "good",
        "VeryMildDemented": "VeryMildDemented",
    }

    def get_label(self, img_path: str) -> list:
        """Get label for file."""
        filename_source = os.path.basename(img_path).split("_")[-1].replace(".png", ".jpg")
        file_source_path = glob.glob(os.path.join(self.args["source_path"], f"**/{filename_source}"), recursive=True)[0]
        return [self.labels_dict[os.path.basename(os.path.dirname(file_source_path))]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
