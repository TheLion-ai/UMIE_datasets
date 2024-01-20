"""Preprocessing pipeline for Alzheimers dataset."""
import fnmatch
import glob
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, List

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
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name=None,
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        replace_dic = {"(": "", ")": "", " ": ""}
        img_id = os.path.basename(img_path)
        for char in replace_dic.keys():
            img_id = img_id.replace(char, replace_dic[char])
        return img_id

    # Changing labels from dataset (folders names) to match standard
    labels_dict = {
        "MildDemented": "MildDemented",
        "ModerateDemented": "ModerateDemented",
        "NonDemented": "good",
        "VeryMildDemented": "VeryMildDemented",
    }
    files_source: list[str] = field(default_factory=list)

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        filename_source = os.path.basename(img_path).split("_")[-1].replace(".png", ".jpg")
        file_source_path = [path for path in self.files_source if filename_source == os.path.basename(path)][0]
        return [self.labels_dict[os.path.basename(os.path.dirname(file_source_path))]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.img_id_extractor(x).replace(".png", "")
        # List of paths to files in source directory with changed names. It will be later used to get labels.
        self.files_source = glob.glob(os.path.join(self.args["source_path"], "**"), recursive=True)
        self.files_source = [name.replace("(", "").replace(")", "").replace(" ", "") for name in self.files_source]

        self.dataset_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
