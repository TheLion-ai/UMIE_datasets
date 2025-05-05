"""Preprocessing pipeline for Amos22 dataset."""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, amos22
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
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Amos22 dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        # Image id is in the source file name after the last underscore
        return self._extract_by_separator(img_path, "_")


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Amos22 dataset."""

    def _extract(self, img_path: str) -> str:
        """Get study ID for dataset."""
        # Study id is the folder name of all images in the study
        filename = os.path.basename(img_path)
        match = re.search(r"_(\d+)_", filename)
        if match is None:
            return ""
        return match.group(1)


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Amos22 dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return "images" in path


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Amos22 dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


@dataclass
class Amos22Pipeline(BasePipeline):
    """Preprocessing pipeline for Amos22 dataset."""

    name: str = "amos22"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("convert_nii2png", ConvertNii2Png),
        ("copy_masks", CopyMasks),
        ("add_umie_ids", AddUmieIds),
        ("recolor_masks", RecolorMasks),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: amos22)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=2,
            img_id_extractor=ImgIdExtractor(),  #
            study_id_extractor=StudyIdExtractor(),
            window_center=50,
            window_width=200,
            img_prefix="",  # prefix of the source image file names
            segmentation_prefix="labels",  # prefix of the source mask file names
            mask_folder_name=MASK_FOLDER_NAME,
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
            nii_slice_by_axis=2,
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Update args with pipeline_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
