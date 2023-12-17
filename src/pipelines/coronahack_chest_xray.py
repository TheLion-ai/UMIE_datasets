"""Preprocessing pipeline for Coronahack Chest XRay dataset."""
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import pandas
import yaml
from sklearn.pipeline import Pipeline

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.create_masks_from_xml import CreateMasksFromXML
from src.steps.get_file_paths import GetFilePaths


@dataclass
class CoronahackChestXrayPipeline(BasePipeline):
    """Preprocessing pipeline for Coronahack Chest XRay dataset."""

    name: str = field(default="CoronaHack_Chest_X-Ray_Dataset")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            # add_new_ids is used here to also add labels
            ("add_new_ids", AddNewIds),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            zfill=4,
            phase_extractor=lambda x: "0",  # All images are from the same phase
        )
    )

    def get_img_id_coronahack(self, img_path: os.PathLike, add_label: bool) -> str | None:
        """Get image name based on its path.

        Args:
            img_path (str): Path to the image.
            add_label (bool): If True returns whole image name with label,
                              if False returns just the image id.
        Returns:
            str: Id of image with, or without labels.
        """
        img_name = os.path.split(img_path)[-1]
        img_row = self.metadata.loc[self.metadata["X_ray_image_name"] == img_name]

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

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()

        # Read metadata csv
        metadata_csv_path = os.path.join(self.args["source_path"], "Chest_xray_Corona_Metadata.csv")
        self.metadata = pandas.read_csv(metadata_csv_path)
        self.metadata.rename(columns={"Unnamed: 0": "id"}, inplace=True)

        self.dataset_args.img_id_extractor = lambda x: self.get_img_id_coronahack(x, True)
        self.dataset_args.study_id_extractor = lambda x: self.get_img_id_coronahack(x, False)

        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
