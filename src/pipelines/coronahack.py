"""Preprocessing pipeline for Coronahack dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, coronahack
from steps import (
    AddLabels,
    AddUmieIds,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    GetFilePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Coronahack dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return "0.png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Coronahack dataset."""

    def __init__(self, labels_path: os.PathLike):
        """Initialize study id extractor."""
        self.metadata = pd.read_csv(labels_path)
        self.metadata.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        img_name = os.path.split(img_path)[-1]
        img_row = self.metadata.loc[self.metadata["X_ray_image_name"] == img_name]

        if img_row.empty or img_name.endswith("csv"):
            # File not present in csv, or is csv
            return ""

        return img_row["id"].values[0]


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Coronahack dataset."""

    def __init__(self, labels: dict[str, list[int]], labels_path: os.PathLike):
        """Initialize label extractor."""
        super().__init__(labels)
        self.source_labels = pd.read_csv(labels_path)
        self.source_labels.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    def _extract(self, img_path: os.PathLike, *args: Any) -> list[dict[str, int]]:
        """Extract label from img path."""
        img_name = os.path.split(img_path)[-1]
        study_id = img_name.split("_")[2]
        img_row = self.source_labels.loc[self.source_labels["id"] == int(study_id)]
        label = img_row["Label"].values[0]

        radlex_labels = []
        if label == "Pnemonia":
            if img_row["Label_1_Virus_category"].values[0] == "bacteria":
                label = "PneumoniaBacteria"
            elif img_row["Label_1_Virus_category"].values[0] == "Virus":
                label = "PneumoniaVirus"

        if label in self.labels.keys():
            radlex_labels = self.labels[label]

        return radlex_labels


@dataclass
class CoronaHackPipeline(BasePipeline):
    """Preprocessing pipeline for Coronahack Chest XRay dataset."""

    name: str = "coronahack"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        # add_new_ids is used here to also add labels
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: coronahack)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            img_id_extractor=ImgIdExtractor(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"], self.args["labels_path"])
        self.args["study_id_extractor"] = StudyIdExtractor(self.args["labels_path"])
