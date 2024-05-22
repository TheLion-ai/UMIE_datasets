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
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class KneeOsteoarthritisPipeline(BasePipeline):
    """Preprocessing pipeline for Knee Osteoarthritis dataset."""

    name: str = field(default="Knee_Osteoarthritis")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("add_new_ids", AddNewIds),
            ("add_new_ids", AddLabels),
            ("delete_temp_files", DeleteTempFiles),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name=None,
        )
    )

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        # replace letters and delete underscore from filenames
        # letters can't be deleted because they make names unique
        study_id = os.path.splitext(os.path.basename(img_path))[0].replace("R", "0").replace("L", "1").replace("_", "")
        study_id = study_id + os.path.basename(os.path.dirname(img_path))
        return study_id

    def img_id_extractor(self, img_path: str) -> str:
        """Each study has only 1 image."""
        return "0"

    # Changing labels from dataset (folders names) to match standard
    labels_dict = {
        "0": "good",
        "1": "DoubtfulOsteoarthritis",
        "2": "MinimalOsteoarthritis",
        "3": "ModerateOsteoarthritis",
        "4": "SevereOsteoarthritis",
    }

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        class_key = os.path.basename(os.path.dirname(img_path))
        return [self.labels_dict[class_key]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        self.dataset_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
