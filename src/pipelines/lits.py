"""Preprocessing pipeline for Liver and liver tumor dataset."""
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2

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

    dataset_args: DatasetArgs = lits
    pipeline_args: PipelineArgs = PipelineArgs(
        img_prefix="volume",  # prefix of the source image file names
        mask_selector="segmentation",
        segmentation_prefix="segmentation",
        multiple_masks_selector={"livermask": "liver", "lesionmask": "liver_tumor"},
        phase_extractor=lambda x: "0",
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        basename = os.path.basename(img_path)  # .split(".")[0]
        img_id = basename.rsplit("_", 1)[1]
        return img_id

    def study_id_extractor(self, img_path: str) -> str:
        """Retrieve study id from path."""
        basename = os.path.basename(img_path).split(".")[0]
        study_id = basename.rsplit("_", 1)[0].rsplit("-", 1)[1]
        return study_id

    def get_label(self, img_path: str) -> list:
        """Get image label based on path."""
        mask_path = img_path.replace(IMG_FOLDER_NAME, MASK_FOLDER_NAME)
        mask = cv2.imread(mask_path)
        if self.args["masks"]["Neoplasm"]["target_color"] in mask:
            return self.args["labels"]["Neoplasm"]
        else:
            return self.args["labels"]["NormalityDescriptor"]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        # Add get_label function to the dataset_args
        self.pipeline_args.get_label = partial(self.get_label)
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
