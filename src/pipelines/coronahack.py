"""Preprocessing pipeline for Coronahack dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, coronahack
from steps import (
    AddLabels,
    AddUmieIds,
    CreateFileTree,
    DeleteImgsWithNoAnnotations,
    GetFilePaths,
)


@dataclass
class CoronaHackPipeline(BasePipeline):
    """Preprocessing pipeline for Coronahack Chest XRay dataset."""

    name: str = "coronahack"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        # add_new_ids is used here to also add labels
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: coronahack)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4, phase_extractor=lambda x: "0", mask_folder_name=None  # All images are from the same phase
        )
    )

    def get_img_id(self, img_path: os.PathLike) -> str | None:
        """Get image name based on its path.

        Args:
            img_path (str): Path to the image.
            add_extension (bool): If True returns image id with *.png extension,
                                  if False returns just the image id.
        Returns:
            str: Id of image with, or without extension.
        """
        img_name = os.path.split(img_path)[-1]
        img_row = self.metadata.loc[self.metadata["X_ray_image_name"] == img_name]

        if img_row.empty or img_name.endswith("csv"):
            # File not present in csv, or is csv
            return None

        return img_row["id"].values[0]

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
        study_id = img_name.split("_")[2]
        img_row = self.metadata.loc[self.metadata["id"] == int(study_id)]
        label = img_row["Label"].values[0]

        radlex_labels = []
        if label == "Pnemonia":
            if img_row["Label_1_Virus_category"].values[0] == "bacteria":
                label = "PneumoniaBacteria"
            elif img_row["Label_1_Virus_category"].values[0] == "Virus":
                label = "PneumoniaVirus"

        if label in self.args["labels"].keys():
            radlex_labels = self.args["labels"][label]

        return radlex_labels

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Read metadata csv
        metadata_csv_path = os.path.join(self.args["source_path"], "Chest_xray_Corona_Metadata.csv")
        self.metadata = pd.read_csv(metadata_csv_path)
        self.metadata.rename(columns={"Unnamed: 0": "id"}, inplace=True)

        self.pipeline_args.img_id_extractor = lambda x: "0.png"
        self.pipeline_args.study_id_extractor = lambda x: self.get_img_id(x)
        self.pipeline_args.get_label = self.get_label

        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
