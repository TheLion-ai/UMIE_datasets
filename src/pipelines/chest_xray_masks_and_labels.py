"""Preprocessing pipeline for lung segmentation from Chest-Xray dataset."""

import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from src.constants import MASK_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.copy_masks import CopyMasks
from src.steps.create_blank_masks import CreateBlankMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths
from src.steps.join_masks import JoinMasks


@dataclass
class LungSegmentationPipeline(BasePipeline):
    """Preprocessing pipeline for Lung_segmentation_from_Chest_X-Rays dataset."""

    name: str = field(default="Lung_segmentation_from_Chest_X-Rays")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("join_masks", JoinMasks),
            ("get_file_paths", GetFilePaths),
            ("copy_masks", CopyMasks),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            ("create_blank_masks", CreateBlankMasks),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            zfill=2,
            # Image id is in the source file name after the last underscore
            img_id_extractor=lambda x: f"0_{x.split('/')[-1].split('_')[-1]}",  #
            # Study id is the folder name of all images in the study
            study_id_extractor=lambda x: x.split("/")[-1].split("_")[-2],
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name="Masks",
            img_prefix="MCUCXR",  # prefix of the source image file names
            segmentation_prefix="Masks",  # prefix of the source mask file names
        )
    )

    def get_label(
        cls,
        img_path: str,
    ) -> list:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            list: List of labels for specific image.
        """
        img_id = os.path.basename(img_path)
        if img_id.split(".")[0].endswith("0"):
            return ["good"]
        return ["tuberculosis"]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Load labels from the labels file
        # good: 0 and 1 as tuberculosis: 27
        self.labels_list = ["good", "tuberculosis"]
        # Add get_label function to the dataset_args
        self.dataset_args.get_label = partial(self.get_label)
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
