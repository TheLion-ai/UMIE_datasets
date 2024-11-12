"""Preprocessing pipeline for Finding_and_Measuring_Lungs_in_CT_Data dataset."""
from dataclasses import asdict, dataclass, field
from operator import index
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, finding_and_measuring_lungs
from constants import IMG_FOLDER_NAME, MASK_FOLDER_NAME
from steps import (
    AddUmieIds,
    ConvertTif2Png,
    CopyMasks,
    CreateFileTree,
    DeleteTempFiles,
    GetFilePaths,
    RecolorMasks,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Finding_and_Measuring_Lungs_in_CT_Data dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._return_zero(suffix=".png")


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Finding_and_Measuring_Lungs_in_CT_Data dataset."""

    def _extract(self, img_path: str) -> str:
        """Get study ID for dataset."""
        # Getting study id depends on location of the file.
        # Study_id is retrieved in a different way when image already is moved to target directory with new name.
        return self._extract_by_separator(img_path, separator="_", index=-3)


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Finding_and_Measuring_Lungs_in_CT_Data dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return "images" in path


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Finding_and_Measuring_Lungs_in_CT_Data dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return "2d_masks" in path


@dataclass
class FindingAndMeasuringLungsPipeline(BasePipeline):
    """Preprocessing pipeline for finding_and_measuring_lungs dataset."""

    name: str = "finding_and_measuring_lungs"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("convert_tif2png", ConvertTif2Png),
        ("copy_png_masks", CopyMasks),
        ("recolor_masks", RecolorMasks),
        ("add_new_ids", AddUmieIds),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: finding_and_measuring_lungs)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name=IMG_FOLDER_NAME,
            mask_folder_name=MASK_FOLDER_NAME,
            img_prefix="images",  # prefix of the source image file names
            segmentation_prefix="masks",  # prefix of the source mask file names
            mask_prefix="2d_masks",
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
