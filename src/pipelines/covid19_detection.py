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
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class Covid19Detection(BasePipeline):
    """Preprocessing pipeline for Alzheimers dataset."""

    name: str = field(default="Covid19_Detection")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("convert_jpg2png", ConvertJpg2Png),
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
    ids_dict_val = {
        "Normal": "0",
        "BacterialPneumonia": "1",
        "ViralPneumonia": "2",
        "COVID-19": "3",
    }
    ids_dict_non_aug = {
        "Normal": "4",
        "BacterialPneumonia": "5",
        "ViralPneumonia": "6",
        "COVID-19": "7",
    }

    def get_study_id(self, img_path: str) -> str:
        """Get study id with added postfix depending on source location to prevent repeated names."""
        img_id = os.path.basename(img_path)
        img_basename = os.path.splitext(img_id)[0]
        if "ValData" in img_basename:
            study_id = img_basename + self.ids_dict_val[os.path.basename(os.path.dirname(img_path))]
        elif "NonAugmentedTrain" in img_basename:
            study_id = img_basename + self.ids_dict_non_aug[os.path.basename(os.path.dirname(img_path))]
        else:
            study_id = img_basename
        return study_id

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        if self.path_args["source_path"] in img_path:
            return self.get_study_id(img_path)
        if self.path_args["target_path"] in img_path:
            img_id = os.path.basename(img_path)
            img_basename = os.path.splitext(img_id)[0]
            img_id = img_basename[:-1]
            return img_id
        return ""

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if self.path_args["source_path"] in img_path:
            img_id = "0"
            img_id = self.study_id_extractor(img_path) + img_id
            return img_id
        if self.path_args["target_path"] in img_path:
            img_id = "0"
            return img_id
        return ""

    # Changing labels from dataset (folders names) to match standard
    labels_dict = {
        "Normal": "good",
        "BacterialPneumonia": "PneumoniaBacteria",
        "ViralPneumonia": "PneumoniaVirus",
        "COVID-19": "covid19",
        "OversampledAugmentedCOVID-19": "covid19",
    }

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        return [self.labels_dict[os.path.basename(os.path.dirname(img_path))]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        self.dataset_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
