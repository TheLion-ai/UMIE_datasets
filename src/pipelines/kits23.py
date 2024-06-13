"""Preprocessing pipeline for KITS23 dataset."""

import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from base.pipeline import BasePipeline, PipelineArgs
from config import dataset_config
from config.dataset_config import DatasetArgs
from constants import MASK_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertNii2Png,
    CopyMasks,
    CreateBlankMasks,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    DeleteTempPng,
    GetFilePaths,
    RecolorMasks,
)


@dataclass
class KITS23Pipeline(BasePipeline):
    """Preprocessing pipeline for KITS23 dataset."""

    name: str = "kits23"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("convert_nii2png", ConvertNii2Png),
        ("copy_masks", CopyMasks),
        ("add_new_ids", AddUmieIds),
        ("recolor_masks", RecolorMasks),
        ("add_labels", AddLabels),
        # Choose either to create blank masks or delete images without masks
        ("create_blank_masks", CreateBlankMasks),
        # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = dataset_config.kits23
    pipeline_args: PipelineArgs = PipelineArgs(
        zfill=2,
        # Image id is in the source file name after the last underscore
        img_id_extractor=lambda x: os.path.basename(x).split("_")[-1],  #
        # Study id is the folder name of all images in the study
        study_id_extractor=lambda x: os.path.basename((os.path.dirname(x))).split("_")[-1],
        window_center=50,  # Window of abddominal cavity CTs
        window_width=400,
        img_prefix="imaging",  # prefix of the source image file names
        segmentation_prefix="segmentation",  # prefix of the source mask file names
        mask_folder_name=MASK_FOLDER_NAME,
    )

    def get_label(
        self,
        img_path: str,
        labels_list: list,
    ) -> list:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            list: List of labels for specific image.
        """
        img_id = os.path.basename(img_path)
        root_path = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(root_path, MASK_FOLDER_NAME, img_id)
        mask = cv2.imread(mask_path)

        kidney_findings_colors = [
            self.args["masks"]["Neoplasm"]["target_color"],
            self.args["masks"]["RenalCyst"]["target_color"],
        ]
        # Check if the mask contains the kidney tumor or cyst
        if np.any(np.isin(kidney_findings_colors, np.unique(mask))):
            # Study id is between the second and third underscore in the target image id
            study_id_regex = re.match(r"^(?:[^_]*_){2}([^_]+)", img_id)
            study_id = (
                study_id_regex.group(1) if study_id_regex is not None else None
            )  # Study id is between the second and third underscore
            labels = []
            for case in labels_list:
                # Find the case with the matching study id
                if case["case_id"] == f"case_{study_id}":
                    # Remove underscores from the label
                    label = case["tumor_histologic_subtype"]
                    # We do not include vague labels
                    if label in self.args["labels"].keys():
                        labels = self.args["labels"]
                    break
            return labels
        return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Load labels from the labels file
        self.labels_list = self.load_labels_from_path(self.args["labels_path"])
        # Add get_label function to the pipeline_args
        self.pipeline_args.get_label = partial(
            self.get_label,
            labels_list=self.labels_list,
        )
        # Update args with pipeline_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
