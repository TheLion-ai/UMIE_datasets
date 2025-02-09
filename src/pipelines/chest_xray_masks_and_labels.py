"""Preprocessing pipeline for ChestX-ray masks and labels dataset."""
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
from config.dataset_config import DatasetArgs, chest_xray_masks_and_labels
from steps import (
    AddLabels,
    AddUmieIds,
    CopyMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    DeleteTempFiles,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Chest Xray masks and labels dataset."""

    def _extract(self, img_path: os.PathLike) -> str:
        """Retrieve image id from path."""
        # Study id is the second part of the image name before the first underscore
        return "0.png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Chest Xray masks and labels dataset."""

    def _extract(self, img_path: os.PathLike) -> str:
        """Extract study id from img path."""
        # Study id is the first part of the image name before the first underscore
        filename = os.path.basename(img_path)
        elements = filename.split("_", 2)
        result = "_".join(elements[:2])
        return result


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Chest Xray masks and labels dataset."""

    def __init__(self, labels: dict[str, str], labels_path: os.PathLike):
        """Initialize the extractor."""
        super().__init__(labels)

    def _extract(self, img_path: os.PathLike, *args: Any) -> tuple[list, list]:
        """Extract label from img path."""
        labels_path = str(img_path).replace("CXR_png", "ClinicalReadings").replace(".png", ".txt")
        label = None

        with open(labels_path, "r") as file:
            lines = file.readlines()
            for line in reversed(lines):
                if line.strip():  # Check if the line is non-empty
                    label = line.strip()
                    break
        radlex_label = None
        source_label = None
        for source_label in self.labels.keys():
            if label is not None and source_label in label:
                radlex_label = self.labels[source_label]
        return [radlex_label], [source_label]


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Chest Xray masks and labels dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        if ".png" in path and "mask" not in path.split("/")[-1] and "CXR_png" in path.split("/")[-2]:
            return True
        else:
            return False


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Chest Xray masks and labels dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        if "/" not in path or "mask" in path.split("/")[-2]:
            return True
        else:
            return False


@dataclass
class ChestXrayMasksAndLabelsPipeline(BasePipeline):
    """Preprocessing pipeline for Chest Xray masks and labels dataset."""

    name: str = "chest_xray_masks_and_labels"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("copy_masks", CopyMasks),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: chest_xray_masks_and_labels)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            study_id_extractor=StudyIdExtractor(),
            mask_prefix="mask",
            segmentation_prefix="mask",
            img_id_extractor=ImgIdExtractor(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"], self.args["labels_path"])
