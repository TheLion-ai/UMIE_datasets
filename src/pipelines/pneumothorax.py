"""Preprocessing pipeline for Pneumothorax dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import (
    BaseImgIdExtractor,
    BaseLabelExtractor,
    BaseStudyIdExtractor,
    study_id,
)
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, siim_acr_pneumothorax
from steps import (
    AddLabels,
    AddUmieIds,
    CopyMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    DeleteTempFiles,
    GetFilePaths,
    StoreSourcePaths,
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Pneumothorax dataset."""

    def _extract(self, img_path: os.PathLike) -> str:
        """Retrieve image id from path."""
        # Study id is the second part of the image name before the first underscore
        return "0" + ".png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Pneumothorax dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study id is the first part of the image name before the first underscore
        train_test_to_unique = {"train": "0", "test": "1"}
        additional_id = train_test_to_unique[self._extract_filename(img_path).split("_")[1]]
        return self._extract_filename(img_path).split("_")[0] + additional_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Pneumothorax dataset."""

    def _extract(self, img_path: os.PathLike, *args: Any) -> tuple[list, list]:
        """Extract label from img path."""
        img_name = os.path.basename(img_path)
        pneumotrax_label = img_name.split("_")[-1][0] == "1"
        if pneumotrax_label:
            labels = ["Pneumothorax"]
        else:
            labels = ["NormalityDecriptor"]
        radlex_labels: list = []
        for label in labels:
            radlex_labels.extend(self.labels[label])

        return radlex_labels, labels


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Pneumothorax dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return "png_images" in path


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Pneumothorax dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return "png_masks" in path


@dataclass
class PneumothoraxPipeline(BasePipeline):
    """Preprocessing pipeline for Pneumothorax dataset."""

    name: str = "siim_acr_pneumothorax"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("copy_masks", CopyMasks),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_files", DeleteTempFiles),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: siim_acr_pneumothorax)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            segmentation_prefix="png_masks",
            mask_prefix="png_masks",
            img_prefix="png_images",
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
        self.args["label_extractor"] = LabelExtractor(self.args["labels"])
