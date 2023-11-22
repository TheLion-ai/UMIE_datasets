"""Preprocess the Alzheimer's dataset."""
import os

import yaml
from sklearn.pipeline import Pipeline

from src.preprocessing.add_labels import AddLabels
from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_dcm2png import ConvertDcm2Png
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.create_masks_from_xml import CreateMasksFromXML
from src.preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from src.preprocessing.get_file_paths import GetFilePaths


def preprocess_alzheimers(source_path: str, target_path: str) -> None:
    """Preprocess the Alzheimer's dataset.

    This function preprocesses the Alzheimer's dataset. It changes names and directory structure to meet standard.
    Args:
        source_path (str): path to the downloaded dataset. Location of the "Alzheimer_s Dataset" folder.
        target_path (str): path to the directory where the preprocessed dataset will be saved.
    """
    dataset_name = "Alzheimers_Dataset"
    dataset_uid = yaml.load(
        open("config/dataset_uid_config.yaml"), Loader=yaml.FullLoader
    )[dataset_name]
    phases = yaml.load(open("config/phases_config.yaml"), Loader=yaml.FullLoader)[
        dataset_name
    ]

    def get_label_alzheimers(img_path: str) -> list:
        return [os.path.basename(os.path.dirname(img_path))]

    params = {
        "source_path": source_path,
        "target_path": target_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "z-fill": 4,
        "img_id_extractor": lambda x: os.path.basename(x),
        "study_id_extractor": lambda x: dataset_name,
        "get_label": get_label_alzheimers,
        "mask_folder_name": "",
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            ("create_file_tree", CreateFileTree(**params)),
            ("add_labels", AddLabels(**params)),
            ("add_new_ids", AddNewIds(**params)),
        ],
    )

    pipeline.transform(X=source_path)
