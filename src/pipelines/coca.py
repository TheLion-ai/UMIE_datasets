"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from config import dataset_config
from src.base.pipeline import BasePipeline, PipelineArgs
from steps.add_umie_ids import AddUmieIds
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.create_masks_from_xml import CreateMasksFromXML
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths


@dataclass
class COCAPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford COCA dataset."""

    name: str = field(default="coca")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("get_file_paths", GetFilePaths),
            ("create_file_tree", CreateFileTree),
            ("convert_dcm2png", ConvertDcm2Png),
            ("create_masks_from_xml", CreateMasksFromXML),
            ("add_new_ids", AddUmieIds),
            # Choose either to create blank masks or delete images without masks
            # ("create_blank_masks", CreateBlankMasks),
            ("delete_imgs_without_masks", DeleteImgsWithNoAnnotations),
            ("delete_temp_png", DeleteTempPng),
        ],
    )
    dataset_args: dataset_config.coca = field(default_factory=lambda: dataset_config.coca)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            # Image id is in the source file name after the last underscore
            img_id_extractor=lambda x: os.path.basename(x).split("-")[-1],
            # Study name is the folder two levels above the image
            study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
