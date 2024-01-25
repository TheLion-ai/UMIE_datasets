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

    ids_dict_test = {
        "NonDemented": "0",
        "VeryMildDemented": "1",
        "MildDemented": "2",
        "ModerateDemented": "3",
    }
    ids_dict_train = {
        "nonDem": "000",
        "verymildDem": "111",
        "mildDem": "222",
        "moderateDem": "333",
    }
    inv_ids_dict_train = {v: k for k, v in ids_dict_train.items()}

    def get_study_id(self, img_path: str) -> str:
        """Retrieve image id from path."""
        basename = os.path.splitext(os.path.basename(img_path))[0]
        if "(" in basename:
            pattern = r"[()]"
            study_id = re.split(pattern, basename)[1]
        else:
            study_id = "0"
        if "train" in img_path:
            for id in self.ids_dict_train.keys():
                basename = basename.replace(id, self.ids_dict_train[id])
            study_id = study_id + basename
        else:
            folder = os.path.basename(os.path.dirname(img_path))
            for id in self.ids_dict_test.keys():
                folder = folder.replace(id, self.ids_dict_test[id])
            study_id = folder + study_id
        return study_id

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        if self.path_args["source_path"] in img_path:
            return self.get_study_id(img_path)
        if self.path_args["target_path"] in img_path:
            img_id = os.path.basename(img_path)
            img_basename = os.path.splitext(img_id)[0]
            img_id = img_basename[:-2]
            return img_id
        return ""

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if self.path_args["source_path"] in img_path:
            img_id = os.path.basename(img_path)
            ext = os.path.splitext(img_id)[1]
            if "train" in img_path:
                img_id = "00"
            else:
                img_id = img_id[:2]
            img_id = img_id + ext
            img_id = self.get_study_id(img_path) + img_id
            return img_id
        if self.path_args["target_path"] in img_path:
            img_id = os.path.basename(img_path)
            img_basename = os.path.splitext(img_id)[0]
            ext = os.path.splitext(img_id)[1]
            img_id = img_basename[-2:] + ext
            return img_id
        return ""

    def reverse_filename(self, img_path: str) -> str:
        """Convert image target name to name in source directory."""
        img_id = os.path.basename(img_path)
        # ext = os.path.splitext(img_id)[1]
        basename = os.path.splitext(img_id)[0]
        study_id = re.split("_", basename)[2]
        img_id = re.split("_", basename)[3]
        filename = ""
        if len(study_id) < 5:
            if study_id[1:] != "0":
                filename = study_id[1:]
                filename = img_id + filename
            else:
                filename = img_id
        else:
            lab = study_id[1:4]
            for id in self.inv_ids_dict_train.keys():
                lab = lab.replace(id, self.inv_ids_dict_train[id])
            filename = lab + study_id[4:]
        filename = filename + ".jpg"
        return filename

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
        # filename_source = os.path.basename(img_path).split("_")[-1].replace(".png", ".jpg")
        filename_source = self.reverse_filename(img_path)
        file_source_path = [path for path in self.files_source if filename_source == os.path.basename(path)][0]
        return [self.labels_dict[os.path.basename(os.path.dirname(file_source_path))]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        # List of paths to files in source directory with changed names. It will be later used to get labels.
        self.files_source = glob.glob(os.path.join(self.args["source_path"], "**"), recursive=True)
        self.files_source = [name.replace("(", "").replace(")", "").replace(" ", "") for name in self.files_source]

        self.dataset_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
