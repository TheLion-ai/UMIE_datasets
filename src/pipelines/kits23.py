"""Preprocessing pipeline for KITS23 dataset."""

import json
import os
import re
from cProfile import label
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, kits23
from constants import MASK_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertNii2Png,
    CopyMasks,
    CreateBlankMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    DeleteTempPng,
    GetFilePaths,
    RecolorMasks,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the KITS23 dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        # Image id is in the source file name after the last underscore
        return os.path.basename(img_path).split("_")[-1]


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the KITS23 dataset."""

    def _extract(self, img_path: str) -> str:
        """Get study ID for dataset."""
        # Study id is the folder name of all images in the study
        return os.path.basename((os.path.dirname(img_path))).split("_")[-1]


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the KITS23 dataset."""

    def __init__(self, labels: dict, labels_path: str, kidney_findings_colors: list):
        """Initialize label extractor."""
        super().__init__(labels)
        with open(labels_path) as f:
            self.labels_list = json.load(f)
        self.kidney_findings_colors = kidney_findings_colors

    def _extract(self, img_path: str, *args: Any) -> list:
        """Extract label from img path."""
        img_id = os.path.basename(img_path)
        root_path = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(root_path, MASK_FOLDER_NAME, img_id)
        mask = cv2.imread(mask_path)

        # Check if the mask contains the kidney tumor or cyst
        if np.any(np.isin(self.kidney_findings_colors, np.unique(mask))):
            # Study id is between the second and third underscore in the target image id
            study_id_regex = re.match(r"^(?:[^_]*_){2}([^_]+)", img_id)
            study_id = (
                study_id_regex.group(1) if study_id_regex is not None else None
            )  # Study id is between the second and third underscore
            labels = []
            for case in self.labels_list:
                # Find the case with the matching study id
                if case["case_id"] == f"case_{study_id}":
                    # Remove underscores from the label
                    source_label = case["tumor_histologic_subtype"]
                    # We do not include vague labels
                    if source_label in self.labels.keys():
                        labels = self.labels[source_label]
                    break
            return labels
        return []


@dataclass
class KITS23Pipeline(BasePipeline):
    """Preprocessing pipeline for KITS23 dataset."""

    name: str = "kits23"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("convert_nii2png", ConvertNii2Png),
        ("copy_masks", CopyMasks),
        ("add_umie_ids", AddUmieIds),
        ("recolor_masks", RecolorMasks),
        ("add_labels", AddLabels),
        # Choose either to create blank masks or delete images without masks
        ("create_blank_masks", CreateBlankMasks),
        # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: kits23)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=2,
            img_id_extractor=ImgIdExtractor(),  #
            study_id_extractor=StudyIdExtractor(),
            window_center=50,  # Window of abddominal cavity CTs
            window_width=400,
            img_prefix="imaging",  # prefix of the source image file names
            segmentation_prefix="segmentation",  # prefix of the source mask file names
            mask_folder_name=MASK_FOLDER_NAME,
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Update args with pipeline_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        kidney_findings_colors = [
            self.args["masks"]["Neoplasm"]["target_color"],
            self.args["masks"]["RenalCyst"]["target_color"],
        ]
        self.args["label_extractor"] = LabelExtractor(
            self.args["labels"], self.args["labels_path"], kidney_findings_colors
        )
