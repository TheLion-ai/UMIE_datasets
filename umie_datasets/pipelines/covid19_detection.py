"""Preprocessing pipeline for Covid 19 detection dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from umie_datasets.base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from umie_datasets.base.pipeline import BasePipeline, PipelineArgs
from umie_datasets.config.dataset_config import DatasetArgs, covid19_detection
from umie_datasets.constants import IMG_FOLDER_NAME
from umie_datasets.steps import (
    AddLabels,
    AddUmieIds,
    ConvertJpg2Png,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Covid 19 detection dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return "0.png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Covid 19 detection dataset."""

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

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
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


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Covid 19 detection dataset."""

    def _extract(self, img_path: str, *args: Any) -> list:
        """Extract label from img path."""
        label = os.path.basename(os.path.dirname(img_path))
        return self.labels[label]


@dataclass
class COVID19DetectionPipeline(BasePipeline):
    """Preprocessing pipeline for Covid 19 detection dataset."""

    name: str = "covid19_detection"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("get_source_paths", StoreSourcePaths),
        ("convert_jpg2png", ConvertJpg2Png),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: covid19_detection)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name=IMG_FOLDER_NAME,
            img_prefix="",
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"])
