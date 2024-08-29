"""Preprocessing pipeline for cmmd dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, cmmd
from constants import IMG_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertDcm2Png,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the cmmd dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        img_id = self._extract_by_separator(img_path, "-")

        # Each image has a name '1-1.dcm' or '1-2.dcm'. This does not
        # ensure uniqueness from the perspective of the entire dataset.
        # Hence in the img_id construction it was decided to use the
        # name of the folder one level higher. From the name in the
        # format '1.000000-NA-xxxxx' (where xxxxx is a series of digits)
        # xxxxx was used. Hence the unique img_id will have the structure
        # xxxxx_1 or xxxxx_2.

        lvl_higher = os.path.basename(os.path.dirname(img_path)).split("-")[2]

        img_id_final = lvl_higher + "_" + img_id 

        return img_id_final


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the cmmd dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study id is supposedly in the folder name two levels above the
        # image name. This folder is written in the format
        # 'mm-dd-yyyy-NA-NA-xxxxx', where xxxxx is a series of numbers
        # representing the study id.

        study_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        study_id = study_id.split("-")[5]

        return study_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the cmmd dataset."""

    def __init__(self, labels: dict[str, list[int]], labels_path: os.PathLike):
        """Initialize label extractor."""
        super().__init__(labels)
        self.source_labels = pd.read_excel(labels_path)
        self.source_labels = self.source_labels[["ID1", "abnormality", "classification"]]
        self.source_labels.rename(columns={"ID1": "id", "abnormality": "label1", "classification": "label2"}, inplace=True)

    def _extract(self, img_path: os.PathLike, *args: Any) -> list[dict[str, int]]:
        """Extract label from img path."""
        img_path_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
        pre_label = self.source_labels[self.source_labels["id"] == img_path_label].values[0]
        label1 = pre_label[1]
        label2 = pre_label[2]

        radlex_label = self.labels[label1] + self.labels[label2]

        return radlex_label


@dataclass
class CmmdPipeline(BasePipeline):
    """Preprocessing pipeline for cmmd dataset."""

    name: str = "cmmd"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("convert_dcm2png", ConvertDcm2Png),
        ("add_umie_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_png", DeleteTempPng),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: cmmd)
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
        self.args["label_extractor"] = LabelExtractor(self.args["labels"], self.args["labels_path"])
