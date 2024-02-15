"""Preprocessing pipeline for Finding_and_Measuring_Lungs_in_CT_Data dataset."""
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_tif2png import ConvertTif2Png
from src.steps.copy_masks import CopyMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths
from src.steps.recolor_masks import RecolorMasks


@dataclass
class FindingAndMeasuringLungsInCTPipeline(BasePipeline):
    """Preprocessing pipeline for Finding_and_Measuring_Lungs_in_CT_Data dataset."""

    name: str = field(default="Finding_and_Measuring_Lungs_in_CT_Data")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("copy_png_masks", CopyMasks),
            ("convert_tif2png", ConvertTif2Png),
            ("recolor_masks", RecolorMasks),
            ("add_new_ids", AddNewIds),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            # Study id is the folder name of all images in the study
            study_id_extractor=lambda x: os.path.basename((os.path.dirname(os.path.dirname(x)))).split("_")[-1],
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name="Masks",
            img_prefix="images",  # prefix of the source image file names
            segmentation_prefix="masks",  # prefix of the source mask file names
            mask_selector="2d_masks",
        )
    )

    def study_id_extractor(self, img_path: str) -> str:
        """Get study ID for dataset."""
        if self.path_args["source_path"] in img_path or self.path_args["masks_path"] in img_path:
            return os.path.basename(img_path).split("_")[-3]
        if self.path_args["target_path"] in img_path:
            return os.path.basename(img_path)[:4]
        return ""

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if self.path_args["source_path"] in img_path or self.path_args["masks_path"] in img_path:
            return os.path.basename(img_path).split("_")[-3] + os.path.basename(img_path).split("_")[-1]
        if self.path_args["target_path"] in img_path:
            return "0"
        return ""

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
