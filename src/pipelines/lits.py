"""Preprocessing pipeline for Liver and liver tumor dataset."""
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, lits
from constants import IMG_FOLDER_NAME, MASK_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    CombineMultipleMasks,
    CopyMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    GetFilePaths,
    RecolorMasks,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Liver and liver tumor dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        basename = os.path.basename(img_path)  # .split(".")[0]
        img_id = basename.rsplit("_", 1)[1]
        return img_id


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Liver and liver tumor dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve study id from path."""
        basename = os.path.basename(img_path).split(".")[0]
        study_id = basename.rsplit("_", 1)[0].rsplit("-", 1)[1]
        return study_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Liver and liver tumor dataset."""

    def __init__(self, labels: dict, target_color: list):
        """Initialize label extractor."""
        super().__init__(labels)
        self.target_color = target_color

    def _extract(self, img_path: str, mask_path: str, *args: Any) -> list:
        """Get image label based on path."""
        mask = cv2.imread(mask_path)
        if mask is not None and (mask == self.target_color).all(axis=-1).any():
            return self.labels["Neoplasm"]
        else:
            return self.labels["NormalityDescriptor"]


@dataclass
class LITSPipeline(BasePipeline):
    """Preprocessing pipeline for Liver and liver tumor dataset."""

    name: str = "Liver_And_Liver_Tumor"  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("combine_multiple_masks", CombineMultipleMasks),
        ("copy_masks", CopyMasks),
        ("recolor_masks", RecolorMasks),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        # Recommended to delete images without masks, because they contain neither liver nor tumor
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
    )

    dataset_args: DatasetArgs = field(default_factory=lambda: lits)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            img_prefix="volume",  # prefix of the source image file names
            mask_selector="segmentation",
            segmentation_prefix="segmentation",
            multiple_masks_selector={"livermask": "liver", "lesionmask": "liver_tumor"},
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(
            self.args["labels"], self.args["masks"]["Neoplasm"]["target_color"]
        )
