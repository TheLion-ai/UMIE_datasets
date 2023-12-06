"""Preprocessing pipeline for KITS21 dataset."""
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np

from config import mask_encodings_config
from src.constants import MASK_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_nii2png import ConvertNii2Png
from src.steps.copy_png_masks import CopyPNGMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.get_file_paths import GetFilePaths
from src.steps.recolor_masks import RecolorMasks


@dataclass
class KITS21Pipeline(BasePipeline):
    """Preprocessing pipeline for KITS21 dataset."""

    name: str = field(default="KITS21")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("convert_nii2png", ConvertNii2Png),
            ("copy_png_masks", CopyPNGMasks),
            ("add_new_ids", AddNewIds),
            ("recolor_masks", RecolorMasks),
            ("add_labels", AddLabels),
            # Choose either to create blank masks or delete images without masks
            # ("create_blank_masks", CreateBlankMasks(**kwargs)),
            ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ]
    )
    dataset_args: DatasetArgs = DatasetArgs(
        zfill=2,
        # Image id is in the source file name after the last underscore
        img_id_extractor=lambda x: os.path.basename(x).split("_")[-1],  #
        # Study id is the folder name of all images in the study
        study_id_extractor=lambda x: os.path.basename((os.path.dirname(x))).split("_")[-1],
        # Windows of abddominal cavity CTs
        window_center=50,
        window_width=400,
        img_dcm_prefix="imaging",  # prefix of the source image file names
        segmentation_dcm_prefix="segmentation",  # prefix of the source mask file names
    )

    def get_label(
        cls,
        img_path: str,
        labels_list: list,
        kidney_tumor_encoding: int = 1,
        mask_folder_name: str = MASK_FOLDER_NAME,
    ) -> list:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            list: List of labels.
        """
        img_id = os.path.basename(img_path)
        root_path = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(root_path, mask_folder_name, img_id)
        mask = cv2.imread(mask_path)

        # Check if the mask contains the kidney tumor
        if kidney_tumor_encoding in np.unique(mask):
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
                    label = case["tumor_histologic_subtype"].replace("_", "")
                    labels.append(label)
                    break
            return labels
        return []

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        kidney_tumor_encoding = mask_encodings_config[self.name]["kidney_tumor"]
        # Load labels from the labels file
        self.labels_list = self.load_labels_from_path()
        # Add get_label function to the dataset_args
        self.dataset_args.get_label = partial(
            self.get_label, kidney_tumor_encoding=kidney_tumor_encoding, labels_list=self.labels_list
        )
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
