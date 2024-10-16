"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, coca
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CreateFileTree,
    CreateMasksFromXML,
    DeleteImgsWithNoAnnotations,
    DeleteTempPng,
    GetFilePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        # Image id is in the source file name after the last underscore
        return self._extract_by_separator(img_path, "-")


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study name is the folder two levels above the image
        return os.path.basename(os.path.dirname(os.path.dirname(img_path)))


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Brain Tumor Progression dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Brain Tumor Progression dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


@dataclass
class COCAPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford COCA dataset."""

    name: str = "coca"  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("convert_dcm2png", ConvertDcm2Png),
        ("create_masks_from_xml", CreateMasksFromXML),
        ("add_new_ids", AddUmieIds),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: coca)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            # Image id is in the source file name after the last underscore
            img_id_extractor=ImgIdExtractor(),
            # Study name is the folder two levels above the image
            study_id_extractor=StudyIdExtractor(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
