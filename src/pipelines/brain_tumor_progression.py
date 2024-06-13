"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, brain_tumor_progression
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CopyMasks,
    CreateFileTree,
    DeleteImgsWithEmptyMasks,
    DeleteTempPng,
    GetFilePaths,
    RecolorMasks,
)


def img_id_extractor(img_path: str) -> str:
    """Retrieve image id from path."""
    return os.path.basename(img_path)


@dataclass
class BrainTumorProgressionPipeline(BasePipeline):
    """Preprocessing pipeline for the Brain Tumor Progression dataset."""

    name: str = field(default="brain_tumor_progression")  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("convert_dcm2png", ConvertDcm2Png),
        ("copy_png_masks", CopyMasks),
        ("recolor_masks", RecolorMasks),
        ("add_umie_ids", AddUmieIds),
        # optionally delete images with empty masks
        ("delete_imgs_with_empty_masks", DeleteImgsWithEmptyMasks),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = brain_tumor_progression
    pipeline_args: PipelineArgs = PipelineArgs(
        zfill=4,
        # Image id is in the source file name after the last underscore
        img_id_extractor=lambda x: os.path.basename(x).split("-")[-1],  # lambda x: os.path.basename(x).split("-")[-1],
        # Study name is the folder two levels above the image
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
        mask_selector="MaskTumor",
        segmentation_prefix="MaskTumor",
    )

    def study_id_extractor(self, img_path: str) -> str:
        """Get study ID for dataset."""
        # Getting study id depends on location of the file.
        # Study_id is retrieved in a different way when image already is moved to target directory with new name.
        if self.path_args["source_path"] in img_path or self.path_args["masks_path"] in img_path:
            return os.path.basename(os.path.dirname(os.path.dirname(img_path)))[-5:]
        if self.path_args["target_path"] in img_path:
            return os.path.basename(img_path)[:5]
        return ""

    def phase_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        for phase in self.dataset_args.phases.keys():
            if self.dataset_args.phases[phase] in img_path:
                return phase
        return ""

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        # self.pipeline_args.img_id_extractor = lambda x: img_id_extractor(x)
        self.pipeline_args.phase_extractor = lambda x: self.phase_id_extractor(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
