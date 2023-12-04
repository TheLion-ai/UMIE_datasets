"""Preprocessing pipeline for KITS21 dataset."""
import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from src.constants import MASK_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.preprocessing.add_labels import AddLabels
from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_nii2png import ConvertNii2Png
from src.preprocessing.copy_png_masks import CopyPNGMasks
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.delete_imgs_with_no_annotations import (
    DeleteImgsWithNoAnnotations,
)
from src.preprocessing.get_file_paths import GetFilePaths
from src.preprocessing.recolor_masks import RecolorMasks


@dataclass
class KITS21Pipeline(BasePipeline):
    """Preprocessing pipeline for KITS21 dataset."""

    name: str = field(default="KITS21")
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
        img_id_extractor=lambda x: os.path.basename(x).split("_")[-1],
        study_id_extractor=lambda x: os.path.basename((os.path.dirname(x))).split("_")[-1],
        window_center=50,  # TODO: add to config
        window_width=400,
        img_dcm_prefix="imaging",
        segmentation_dcm_prefix="segmentation",
    )

    def get_label(
        cls,
        img_path: str,
        labels_list: list,
        kidney_tumor_encoding: int = 1,  # = mask_encoding_config["kidney_tumor"],
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

        if kidney_tumor_encoding in np.unique(mask):
            study_id_regex = re.match(r"^(?:[^_]*_){2}([^_]+)", img_id)
            study_id = (
                study_id_regex.group(1) if study_id_regex is not None else None
            )  # Study id is between the second and third underscore
            labels = []
            for case in labels_list:
                if case["case_id"] == f"case_{study_id}":
                    label = case["tumor_histologic_subtype"].replace("_", "")
                    labels.append(label)
                    break
            return labels
        return []

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        self.labels_list = self.load_labels_from_path()
        self.dataset_args.get_label = partial(self.get_label, labels_list=self.labels_list)
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
