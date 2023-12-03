"""Preprocessing pipeline for KITS21 dataset."""
import json
import os
import re

import cv2
import numpy as np
import yaml
from sklearn.pipeline import Pipeline

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


def preprocess_kits21(source_path: str, target_path: str, labels_path: str) -> None:
    """Preprocess the KITS21 dataset.

    Args:
        source_path (str): Path to the source directory.
        target_path (str): Path to the target directory.
        labels_path (str): Path to the labels file.
    """
    dataset_name = "KITS21"
    dataset_uid = yaml.load(
        open("config/dataset_uid_config.yaml"), Loader=yaml.FullLoader
    )[dataset_name]
    phases = yaml.load(open("config/phases_config.yaml"), Loader=yaml.FullLoader)[
        dataset_name
    ]
    mask_encoding_config = yaml.load(
        open("config/masks_encoding_config.yaml"), Loader=yaml.FullLoader
    )
    dataset_masks = yaml.load(
        open("config/dataset_masks_config.yaml"), Loader=yaml.FullLoader
    )[dataset_name]

    mask_colors_old2new = {v: mask_encoding_config[k] for k, v in dataset_masks.items()}

    labels_path_extention = os.path.basename(labels_path).split(".")[1]
    if labels_path_extention == "json":
        with open(labels_path) as f:
            labels_list = json.load(f)

    def get_label_kits19(
        img_path: str,
        mask_folder_name: str = "Masks",
        kidney_tumor_encoding: int = mask_encoding_config["kidney_tumor"],
        labels_list: list = labels_list,
    ) -> list:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            kidney_tumor_encoding (int, optional): Encoding of the kidney tumor mask. Defaults to 1.
            labels_list (list, optional): List of labels. Defaults to labels_list.

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

    params = {
        "source_path": source_path,
        "target_path": target_path,
        "labels_path": labels_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "z-fill": 2,
        "img_id_extractor": lambda x: os.path.basename(x).split("_")[-1],
        "study_id_extractor": lambda x: os.path.basename((os.path.dirname(x))).split(
            "_"
        )[-1],
        "mask_colors_old2new": mask_colors_old2new,
        "window_center": 50,  # TODO: add to config
        "window_width": 400,
        "mask_selector": "segmentation",
        "get_label": get_label_kits19,
        "img_dcm_prefix": "imaging",
        "segmentation_dcm_prefix": "segmentation",
    }

    pipeline = Pipeline(
        steps=[
            ("create_file_tree", CreateFileTree(**params)),
            ("get_file_paths", GetFilePaths(**params)),
            (
                "convert_nii2png",
                ConvertNii2Png(**params),
            ),
            ("copy_png_masks", CopyPNGMasks(**params)),
            ("add_new_ids", AddNewIds(**params)),
            ("recolor_masks", RecolorMasks(**params)),
            ("add_labels", AddLabels(**params)),
            # Choose either to create blank masks or delete images without masks
            # ("create_blank_masks", CreateBlankMasks(**params)),
            ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations(**params)),
        ],
    )

    pipeline.transform(X=source_path)
