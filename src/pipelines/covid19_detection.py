"""Preprocessing pipeline for Covid 19 detection dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from config import dataset_config
from src.constants import IMG_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class COVID19DetectionPipeline(BasePipeline):
    """Preprocessing pipeline for Covid 19 detection dataset."""

    name: str = field(default="covid19_detection")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("convert_jpg2png", ConvertJpg2Png),
            ("add_new_ids", AddNewIds),
            ("add_new_ids", AddLabels),
            ("delete_temp_files", DeleteTempFiles),
            ("delete_temp_png", DeleteTempPng),
        ]
    )
    dataset_args: dataset_config.covid19_detection = field(default_factory=lambda: dataset_config.covid19_detection)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name=IMG_FOLDER_NAME,
            mask_folder_name=None,
            img_prefix="",
        )
    )
    # Images in the dataset has non-unique ids between classes and folders.
    # Dictionaries below are used to make them unique across whole dataset.

    # Ids added to image names in ValData folder based on their classes
    ids_dict_val = {
        "Normal": "000",
        "BacterialPneumonia": "111",
        "ViralPneumonia": "222",
        "COVID-19": "333",
    }
    # Ids added to image names in NonAugmentedTrain folder based on their classes
    ids_dict_non_aug = {
        "Normal": "444",
        "BacterialPneumonia": "555",
        "ViralPneumonia": "666",
        "COVID-19": "777",
    }

    def study_id_extractor(self, img_path: str) -> str:
        """Get study id with added postfix depending on source location to prevent repeated names."""
        img_id = os.path.basename(img_path)
        img_basename = os.path.splitext(img_id)[0]
        if "ValData" in img_path:
            study_id = img_basename + self.ids_dict_val[os.path.basename(os.path.dirname(img_path))]
        elif "NonAugmentedTrain" in img_path:
            study_id = img_basename + self.ids_dict_non_aug[os.path.basename(os.path.dirname(img_path))]
        else:
            study_id = img_basename
        study_id = study_id.replace("_", "")
        return study_id

    def img_id_extractor(self, img_path: str) -> str:
        """In this dataset only one image exists for each study."""
        return "0"

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        label = os.path.basename(os.path.dirname(img_path))
        return self.args["labels"][label]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        self.pipeline_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
