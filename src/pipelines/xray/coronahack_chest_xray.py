import os

import pandas
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
    images to PNG format, and adds lavels to file names.

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

    images_path = os.path.join(source_path, "Coronahack-Chest-XRay-Dataset")

    # Read metadata csv
    metadata_csv_path = os.path.join(source_path, "Chest_xray_Corona_Metadata.csv")
    metadata = pandas.read_csv(metadata_csv_path)
    metadata.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    def get_img_id_coronahack(img_path: os.PathLike, add_label: bool) -> str | None:
        img_name = os.path.split(img_path)[-1]
        img_row = metadata.loc[metadata["X_ray_image_name"] == img_name]

        if img_row.empty:
            # File not present in csv
            return None
        else:
            if not add_label:
                # Used as study_id_extractor
                return img_row["id"].values[0]
            else:
                # Create Labels
                if img_row["Label"].values[0] == "Normal":
                    label = "good"
                elif img_row["Label"].values[0] == "Pnemonia":
                    if img_row["Label_1_Virus_category"].values[0] == "bacteria":
                        label = "PneumoniaBacteria"
                    elif img_row["Label_1_Virus_category"].values[0] == "Virus":
                        label = "PneumoniaVirus"
                    else:
                        return None
                else:
                    return None

            return f'{img_row["id"].values[0]}_{label}.png'

    params = {
        "source_path": images_path,
        "target_path": target_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "z-fill": 4,
        "phases": phases,
        "mask_folder_name": None,
        "img_id_extractor": lambda x: get_img_id_coronahack(x, True),
        "study_id_extractor": lambda x: get_img_id_coronahack(x, False),
    }

    pipeline = Pipeline(
        steps=[
            ("get_file_paths", GetFilePaths(**params)),
            ("create_file_tree", CreateFileTree(**params)),
            # add_new_ids is used here to also add labels
            ("add_new_ids", AddNewIds(**params)),
        ],
    )

    pipeline.transform(X=source_path)
