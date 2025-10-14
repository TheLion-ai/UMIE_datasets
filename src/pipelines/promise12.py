"""Preprocessing pipeline for Promise12 dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, promise12
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertMhd2Png,
    CopyMasks,
    CreateBlankMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    DeleteTempFiles,
    GetFilePaths,
    RecolorMasks,
    StoreSourcePaths,
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Promise12 dataset."""

    def _extract(self, img_path: os.PathLike) -> str:
        """Retrieve image id."""
        img_id = self._extract_by_separator(img_path, "_")
        return img_id


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Promise12 dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study id is the first part of the image name before the first underscore
        study_id = self._extract_filename(img_path).split("_")[0].replace("Case", "")
        if "livechallenge_test_data" in img_path:
            study_id += "0"
        elif "test_data" in img_path:
            study_id += "1"
        elif "training_data" in img_path:
            study_id += "2"
        return study_id


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Promise12 dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return "segmentation" not in path


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Promise12 dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return "segmentation" in path


@dataclass
class Promise12Pipeline(BasePipeline):
    """Preprocessing pipeline for Promise12 dataset."""

    name: str = "promise12"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("convert_mhd2png", ConvertMhd2Png),
        ("copy_png_masks", CopyMasks),
        ("recolor_masks", RecolorMasks),
        ("add_new_ids", AddUmieIds),
        # delete images with empty masks (there shouldn't be any in this dataset)
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_files", DeleteTempFiles),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: promise12)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            segmentation_prefix="segmentation",
            mask_prefix="segmentation",
            img_prefix="",
            study_id_extractor=StudyIdExtractor(),
            img_id_extractor=ImgIdExtractor(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
