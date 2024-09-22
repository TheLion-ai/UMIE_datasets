"""Preprocessing pipeline for CT-ORG dataset."""

import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config import dataset_config
from config.dataset_config import DatasetArgs
from constants import MASK_FOLDER_NAME
from steps import (
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
    """Extractor for image IDs specific to the Knee Osteoarthritis dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract image id from img path."""
        return os.path.basename(img_path).split("_")[-1]


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Knee Osteoarthritis dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        study_id = os.path.basename(img_path).replace("-", "_").split("_")[-2]
        return study_id


@dataclass
class CTORGPipeline(BasePipeline):
    """Preprocessing pipeline for CT-ORG dataset."""

    name: str = "ct_org"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("convert_nii2png", ConvertNii2Png),
        ("copy_masks", CopyMasks),
        ("add_umie_ids", AddUmieIds),
        ("recolor_masks", RecolorMasks),
        # Choose either to create blank masks or delete images without masks
        ("create_blank_masks", CreateBlankMasks),
        # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: dataset_config.ct_org)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=2,
            # Image id is in the source file name after the last underscore
            img_id_extractor=ImgIdExtractor(),  #
            # Study id is the folder name of all images in the study
            study_id_extractor=StudyIdExtractor(),
            window_center=50,  # Window of abddominal cavity CTs
            window_width=400,
            img_prefix="volume",  # prefix of the source image file names
            segmentation_prefix="labels",  # prefix of the source mask file names
            mask_folder_name=MASK_FOLDER_NAME,
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
