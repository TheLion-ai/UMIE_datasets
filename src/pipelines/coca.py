"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, coca
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CreateFileTree,
    CreateMasksFromXML,
    DeleteImgsWithNoAnnotations,
    DeleteTempPng,
    GetFilePaths,
)


@dataclass
class COCAPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford COCA dataset."""

    name: str = "coca"  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("convert_dcm2png", ConvertDcm2Png),
        ("create_masks_from_xml", CreateMasksFromXML),
        ("add_new_ids", AddUmieIds),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks),
        ("delete_imgs_without_masks", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = coca
    pipeline_args: PipelineArgs = PipelineArgs(
        zfill=4,
        # Image id is in the source file name after the last underscore
        img_id_extractor=lambda x: os.path.basename(x).split("-")[-1],
        # Study name is the folder two levels above the image
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
