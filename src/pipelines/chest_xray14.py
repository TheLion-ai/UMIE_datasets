"""Preprocessing pipeline for ChestX-ray14 dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from config import dataset_config
from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.get_file_paths import GetFilePaths


@dataclass
class ChestXray14Pipeline(BasePipeline):
    """Preprocessing pipeline for Chest Xray 14 dataset."""

    name: str = field(default="chest_xray14")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ]
    )
    dataset_args: dataset_config.chest_xray14 = field(default_factory=lambda: dataset_config.chest_xray14)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
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
        labels = [label for label in img_row["Finding Labels"].values[0].split("|")]
        radlex_labels = [self.args["label2radlex"][label] for label in labels]

        return radlex_labels

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Read metadata csv
        metadata_csv_path = self.path_args[
            "labels_path"
        ]  # os.path.join(self.args["source_path"], "Data_Entry_2017_v2020.csv")
        self.metadata = pd.read_csv(metadata_csv_path)

        self.pipeline_args.img_id_extractor = lambda x: self.get_img_id(x, True)
        self.pipeline_args.study_id_extractor = lambda x: self.get_img_id(x, False)
        self.pipeline_args.get_label = self.get_label

        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
