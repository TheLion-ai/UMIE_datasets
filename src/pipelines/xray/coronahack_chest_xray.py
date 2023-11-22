import os

import yaml
from sklearn.pipeline import Pipeline

from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_dcm2png import ConvertDcm2Png
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.create_masks_from_xml import CreateMasksFromXML
from src.preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from src.preprocessing.get_file_paths import GetFilePaths


def preprocess_coronahack_chest_xray(source_path: str, target_path: str) -> None:
    """Preprocess the Coronahack Chest X-Ray dataset.

    This function preprocesses the Coronahack Chest X-Ray dataset. It converts the JPEG
    images to PNG format.

    Args:
        source_path (str): path to the downloaded dataset. Location of the root archive folder
                           (one, that contains csv's and Coronahack-Chest-XRay-Dataset folder).
        target_path (str): path to the directory where the preprocessed dataset will be saved.
    """
    dataset_name = "CoronaHack_Chest_X-Ray_Dataset"
    dataset_uid = yaml.load(
        open("config/dataset_uid_config.yaml"), Loader=yaml.FullLoader
    )[dataset_name]
    phases = yaml.load(open("config/phases_config.yaml"), Loader=yaml.FullLoader)[
        dataset_name
    ]

    params = {
        "source_path": source_path,
        "target_path": target_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            # ("create_file_tree", CreateFileTree(**params)),
            # ("add_new_ids", AddNewIds(**params)),
            # ("convert_dcm2png", ConvertDcm2Png(**params)),
            # ("create_masks_from_xml", CreateMasksFromXML(**params)),
            # # Choose either to create blank masks or delete images without masks
            # # ("create_blank_masks", CreateBlankMasks(**params)),
            # ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params)),
        ],
    )

    pipeline.transform(X=source_path)
