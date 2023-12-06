"""Preprocess the Brain with hemorrhage dataset."""
import os

import yaml
from sklearn.pipeline import Pipeline

from src.preprocessing.add_labels import AddLabels
from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_jpg2png import ConvertJpg2Png
from src.preprocessing.copy_png_masks import CopyPNGMasks
from src.preprocessing.create_blank_masks import CreateBlankMasks
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from src.preprocessing.get_file_paths import GetFilePaths
from src.preprocessing.masks_to_binary_colors import MasksToBinaryColors
from src.preprocessing.recolor_masks import RecolorMasks


def preprocess_brain_with_hemorrhage(
    source_path: str, target_path: str, masks_path: str
) -> None:
    """Preprocess the Brain with hemorrhage dataset.

    This function preprocesses the Brain with hemorrhage dataset. It changes names and directory structure to meet standard.
    Args:
        source_path (str): path to the downloaded dataset. Location of the "Brain with hemorrhage Dataset" folder.
        target_path (str): path to the directory where the preprocessed dataset will be saved.
    """
    dataset_name = "Brain_with_hemorrhage"
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

    def get_label_brain_with_hemorrhage(img_path: str) -> list:
        dir = os.path.dirname(img_path)
        name = os.path.basename(img_path)
        name, ext = os.path.splitext(name)
        mask_path = os.path.join(dir, name + "_HGE_Seg" + ext)
        if os.path.exists(mask_path):
            return ["hemorrhage"]
        else:
            return ["good"]

    def phase_brain_with_hemorrhage(img_path: str) -> str:
        if (
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
            == target_path
        ):
            phase_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        else:
            phase_name = os.path.basename(os.path.dirname(img_path))
        lowercase_phases = [x.lower() for x in list(phases.values())]
        return list(phases.keys())[lowercase_phases.index(phase_name.lower())]

    def img_id_brain_with_hemorrhage(img_path: str) -> str:
        if source_path in img_path:
            return os.path.basename(
                os.path.dirname(os.path.dirname(img_path))
            ) + os.path.basename(img_path)
        if target_path in img_path:
            return os.path.basename(img_path)
        return ""

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
        "img_id_extractor": img_id_brain_with_hemorrhage,
        "study_id_extractor": lambda x: dataset_name,
        "get_label": get_label_brain_with_hemorrhage,
        "mask_colors_old2new": mask_colors_old2new,
        "mask_selector": "_HGE_Seg",
        "phase_extractor": phase_brain_with_hemorrhage,
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            ("create_file_tree", CreateFileTree(**params)),
            ("copy_masks", CopyPNGMasks(**params)),
            ("add_labels", AddLabels(**params)),
            ("add_new_ids", AddNewIds(**params)),
            ("convert_jpg2png", ConvertJpg2Png(**params)),
            ("masks_to_binary_colors", MasksToBinaryColors(**params)),
            ("recolor_masks", RecolorMasks(**params)),
            # Choose either to create blank masks or delete images without masks
            # Recommended to create blank masks because only about 10% images have masks.
            ("create_blank_masks", CreateBlankMasks(**params)),
            # ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params)),
        ],
    )

    pipeline.transform(X=source_path)
