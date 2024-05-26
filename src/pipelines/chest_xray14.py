"""Preprocessing pipeline for ChestX-ray14 dataset."""
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import pandas
import yaml
from sklearn.pipeline import Pipeline

from config.labels import labels
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.create_masks_from_xml import CreateMasksFromXML
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.get_file_paths import GetFilePaths


@dataclass
class ChestXray14Pipeline(BasePipeline):
    """Preprocessing pipeline for Chest Xray 14 dataset."""

    name: str = field(default="ChestX-ray14")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ]
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            zfill=4, phase_extractor=lambda x: "0", mask_folder_name=None  # All images are from the same phase
        )
    )

    def get_img_id(self, img_path: os.PathLike, add_extension: bool) -> str | None:
        """Get image name based on its path.

        Args:
            img_path (str): Path to the image.
            add_extension (bool): If True returns image id with *.png extension,
                                  if False returns just the image id.
        Returns:
            str: Id of image with, or without extension.
        """
        img_name = os.path.split(img_path)[-1]

        img_row = self.metadata.loc[self.metadata["Image Index"] == img_name]

        if img_row.empty or img_name.endswith("csv"):
            # File not present in csv, or is csv
            return None

        if not add_extension:
            # Used as study_id_extractor
            return img_row["Image Index"].values[0].split(".")[0]
        return f'{img_row["Image Index"].values[0]}'

    def get_label(
        self,
        img_path: os.PathLike,
    ) -> list | None:
        """Get label for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            list | None: List of labels for specific image,
                         or None if no are present.
        """
        img_name = os.path.split(img_path)[-1]
        img_id = img_name.split("_")
        img_id = f"{img_id[4]}_{img_id[5]}"
        if ".png" not in img_id:
            img_id += ".png"
        img_row = self.metadata.loc[self.metadata["Image Index"] == img_id]
        found_labels = [label for label in img_row["Finding Labels"].values[0].split("|")]
        for label in found_labels:
            label = "".join(split_label.capitalize() for split_label in label.split("_"))
            if label == "No Findings":
                label = "good"
        return found_labels

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Read metadata csv
        metadata_csv_path = os.path.join(self.args["source_path"], "Data_Entry_2017_v2020.csv")
        self.metadata = pandas.read_csv(metadata_csv_path)

        self.dataset_args.img_id_extractor = lambda x: self.get_img_id(x, True)
        self.dataset_args.study_id_extractor = lambda x: self.get_img_id(x, False)
        self.dataset_args.get_label = self.get_label

        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
