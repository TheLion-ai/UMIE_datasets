"""Preprocessing pipeline for the DBT dataset."""
import os
from dataclasses import dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, dbt
from constants import IMG_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertDcm2Png,
    CreateFileTree,
    DeleteTempPng,
    GetFilePaths,
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the DBT dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._extract_by_separator(img_path, "-")


# TODO: class StudyIDExtractor():


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the DBT dataset."""

    def _extract(self, img_path: str, *args: Any) -> tuple[list, list]:
        """Extract label from img path."""
        source_label = os.path.basename(os.path.dirname(img_path))
        radlex_label = self.labels[source_label]
        return radlex_label, [source_label]


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the DBT dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


@dataclass
class DBTPipeline(BasePipeline):
    """Preprocessing pipeline for the DBT dataset."""

    name: str = "dbt"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("convert_dcm2png", ConvertDcm2Png),
        ("add_umie_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_png", DeleteTempPng),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: dbt)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=2,
            img_id_extractor=ImgIdExtractor(),  #
            # TODO: study_id_extractor=StudyIdExtractor(),
            window_center=50,
            window_width=400,
            img_prefix="imaging",
            segmentation_prefix="segmentation",
            img_selector=ImageSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # TODO: complete
