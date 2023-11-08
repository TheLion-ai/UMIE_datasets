import os

import yaml
from sklearn.pipeline import Pipeline

from preprocessing.add_new_ids import AddNewIds
from preprocessing.copy_png_masks import CopyPNGMasks
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.get_file_paths import GetFilePaths
from preprocessing.recolor_masks import RecolorMasks


def preprocess_stanford_brain_met(source_path: str, target_path: str):
    """Preprocess the Stanford Brain MET dataset.
    Args:
        source_path (str): Path to the source directory.
        target_path (str): Path to the target directory.
    """
    dataset_name = "Stanford_Brain_MET"
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

    params = {
        "target_path": target_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "zfill": 3,
        "study_id_extractor": lambda x: os.path.basename(
            os.path.dirname(os.path.dirname(x))
        ).split("_")[-1],
        "phase_extractor": lambda x: os.path.basename(os.path.dirname(x)),
        "mask_colors_old2new": mask_colors_old2new,
    }

    pipeline = Pipeline(
        steps=[
            ("create_file_tree", CreateFileTree(**params)),
            ("get_file_paths", GetFilePaths(**params)),
            ("copy_png_masks", CopyPNGMasks(**params)),
            ("add_new_ids", AddNewIds(**params)),
            ("recolor_masks", RecolorMasks(**params)),
        ]
    )

    pipeline.transform(X=source_path)
