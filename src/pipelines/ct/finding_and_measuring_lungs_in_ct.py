"""Preprocess the Finding and Measuring Lungs in CT Data dataset."""
import os

import yaml
from sklearn.pipeline import Pipeline

from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_tif2png import ConvertTif2Png
from src.preprocessing.copy_png_masks import CopyPNGMasks
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.get_file_paths import GetFilePaths
from src.preprocessing.recolor_masks import RecolorMasks


def preprocess_lungs_ct(source_path: str, target_path: str, masks_path: str) -> None:
    """Preprocess the Finding and Measuring Lungs in CT Data dataset.

    This function preprocesses the dataset. It converts the tif images to PNG format and modify masks to match encoding.
    Args:
        source_path (str): path to the downloaded dataset. Location of the "2d_images" folder.
        target_path (str): path to the directory where the preprocessed dataset will be saved.
        masks_path (str): path to the directory where the downloaded masks are stored. Location of the "2d_masks" folder.
    """
    dataset_name = "Finding_and_Measuring_Lungs_in_CT_Data"
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
    target_colors = mask_colors_old2new

    params = {
        "source_path": source_path,
        "target_path": target_path,
        "masks_path": masks_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "target_colors": target_colors,
        "z-fill": 4,
        "img_id_extractor": lambda x: os.path.basename(x).split("-")[-1],
        "study_id_extractor": lambda x: os.path.basename(
            os.path.dirname(os.path.dirname(x))
        ),
        "mask_colors_old2new": mask_colors_old2new,
        "mask_selector": "masks",
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            ("create_file_tree", CreateFileTree(**params)),
            ("copy_masks", CopyPNGMasks(**params)),
            ("add_new_ids", AddNewIds(**params)),
            ("convert_tif2png", ConvertTif2Png(**params)),
            ("recolor_masks", RecolorMasks(**params)),
        ],
    )

    pipeline.transform(X=source_path)
