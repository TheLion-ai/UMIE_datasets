"""Preprocessing pipeline for ChestX-ray14 dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, chest_xray14
from steps import (
    AddLabels,
    AddUmieIds,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    GetFilePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Chest Xray 14 dataset."""

    def __init__(self, metadata: pd.DataFrame):
        """Initialize the extractor."""
        super().__init__()
        self.metadata = pd.read_csv(metadata)

    def _extract(self, img_path: os.PathLike) -> str:
        """Retrieve image id from path."""
        img_name = os.path.split(img_path)[-1]

        img_row = self.metadata.loc[self.metadata["Image Index"] == img_name]

        if img_row.empty or img_name.endswith("csv"):
            # File not present in csv, or is csv
            return ""

        return f'{img_row["Image Index"].values[0]}'


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Chest Xray 14 dataset."""

    def __init__(self, metadata: pd.DataFrame):
        """Initialize the extractor."""
        super().__init__()
        self.metadata = pd.read_csv(metadata)

    def _extract(self, img_path: os.PathLike) -> str:
        """Extract study id from img path."""
        img_name = os.path.split(img_path)[-1]

        img_row = self.metadata.loc[self.metadata["Image Index"] == img_name]

        if img_row.empty:
            # File not present in csv, or is csv
            return ""

        # Used as study_id_extractor
        return img_row["Image Index"].values[0].split(".")[0]


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Brain Tumor Detection dataset."""

    def __init__(self, labels: dict[str, str], labels_path: os.PathLike):
        """Initialize the extractor."""
        super().__init__(labels)
        self.source_labels = pd.read_csv(labels_path)[["Image Index", "Finding Labels"]]

    def _extract(self, img_path: os.PathLike, mask_path: os.PathLike) -> list:
        """Extract label from img path."""
        img_name = os.path.split(img_path)[-1]
        img_id = img_name.split("_")
        img_id = f"{img_id[4]}_{img_id[5]}"
        if ".png" not in img_id:
            img_id += ".png"
        img_row = self.source_labels.loc[self.source_labels["Image Index"] == img_id]
        labels = [label for label in img_row["Finding Labels"].values[0].split("|")]
        radlex_labels = [self.labels[label] for label in labels]

        return radlex_labels


@dataclass
class ChestXray14Pipeline(BasePipeline):
    """Preprocessing pipeline for Chest Xray 14 dataset."""

    name: str = "chest_xray14"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
    )
    dataset_args: DatasetArgs = chest_xray14
    pipeline_args: PipelineArgs = PipelineArgs(
        zfill=4,
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["img_id_extractor"] = ImgIdExtractor(self.args["labels_path"])
        self.args["study_id_extractor"] = StudyIdExtractor(self.args["labels_path"])
        self.args["label_extractor"] = LabelExtractor(self.args["labels"], self.args["labels_path"])
