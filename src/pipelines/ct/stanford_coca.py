import os

import yaml
from sklearn.pipeline import Pipeline

from preprocessing.add_new_ids import AddNewIds
from preprocessing.convert_dcm2png import ConvertDcm2Png
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.create_masks_from_xml import CreateMasksFromXML
from preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from preprocessing.get_file_paths import GetFilePaths


def preprocess_coca(source_path: str, target_path: str, masks_path: str):
    """Preprocess the Stanford COCA dataset.
    This function preprocesses the Stanford COCA dataset. It converts the DICOM images to PNG format and creates masks from the XML files.

    Args:
        source_path (str): path to the downloaded dataset. Location of the "patient" folder.
        target_path (str): path to the directory where the preprocessed dataset will be saved.
        masks_path (str): path to the directory where the downloaded masks are stored. Location of the "calcium_xml" folder.
    """

    dataset_name = "Stanford_COCA"
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
        "masks_path": masks_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "z-fill": 4,
        "img_id_extractor": lambda x: os.path.basename(x).split("-")[-1],
        "study_id_extractor": lambda x: os.path.basename(
            os.path.dirname(os.path.dirname(x))
        ),
        "mask_colors_old2new": mask_colors_old2new,
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            ("create_file_tree", CreateFileTree(**params)),
            ("add_new_ids", AddNewIds(**params)),
            ("convert_dcm2png", ConvertDcm2Png(**params)),
            ("create_masks_from_xml", CreateMasksFromXML(**params)),
            # Choose either to create blank masks or delete images without masks
            # ("create_blank_masks", CreateBlankMasks(**params)),
            ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params)),
        ],
    )

    pipeline.transform(X=source_path)
