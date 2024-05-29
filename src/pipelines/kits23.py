"""Preprocessing pipeline for KITS23 dataset."""

import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from src.constants import MASK_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_nii2png import ConvertNii2Png
from src.steps.copy_masks import CopyMasks
from src.steps.create_blank_masks import CreateBlankMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths
from src.steps.recolor_masks import RecolorMasks


@dataclass
class KITS23Pipeline(BasePipeline):
    """Preprocessing pipeline for KITS23 dataset."""

    name: str = field(default="KITS23")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("convert_nii2png", ConvertNii2Png),
            ("copy_masks", CopyMasks),
            ("add_new_ids", AddNewIds),
            ("recolor_masks", RecolorMasks),
            ("add_labels", AddLabels),
            # Choose either to create blank masks or delete images without masks
            ("create_blank_masks", CreateBlankMasks),
            # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
            ("delete_temp_png", DeleteTempPng),
        ]
    )
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=2,
            # Image id is in the source file name after the last underscore
            img_id_extractor=lambda x: os.path.basename(x).split("_")[-1],  #
            # Study id is the folder name of all images in the study
            study_id_extractor=lambda x: os.path.basename((os.path.dirname(x))).split("_")[-1],
            phase_extractor=lambda x: "0",  # All images are from the same phase
            window_center=50,  # Window of abddominal cavity CTs
            window_width=400,
            img_prefix="imaging",  # prefix of the source image file names
            segmentation_prefix="segmentation",  # prefix of the source mask file names
        )
    )

    def get_label(
        cls,
        img_path: str,
        label2radlex: list,
        labels_list: list,
        kidney_findings_encodings: list = [1, 2],
        mask_folder_name: str = MASK_FOLDER_NAME,
    ) -> list:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            list: List of labels for specific image.
        """
        img_id = os.path.basename(img_path)
        root_path = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(root_path, mask_folder_name, img_id)
        mask = cv2.imread(mask_path)

        # Check if the mask contains the kidney tumor or cyst
        if np.any(np.isin(kidney_findings_encodings, np.unique(mask))):
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
                    radlex_labels = label2radlex[label]
                    labels = radlex_labels
                    break
            return labels
        return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        kidney_findings_encodings = [2, 3]
        # Load labels from the labels file
        self.labels_list = self.load_labels_from_path()
        # Add get_label function to the pipeline_args
        self.pipeline_args.get_label = partial(
            self.get_label, label2radlex=self.dataset_args.label2radlex, kidney_findings_encodings=kidney_findings_encodings, labels_list=self.labels_list
        )
        # Update args with pipeline_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
