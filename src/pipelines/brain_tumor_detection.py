"""Preprocessing pipeline for Brain Tumor Detection dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, brain_tumor_detection
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertJpg2Png,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)

ids_dict = {" ": "0", "n": "1", "o": "2", "N": "3", "Y": "4"}


@dataclass
class BrainTumorDetectionPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Detection dataset."""

    name: str = "brain_tumor_detection"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("get_source_paths", StoreSourcePaths),
        ("convert_jpg2png", ConvertJpg2Png),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = brain_tumor_detection
    pipeline_args: PipelineArgs = PipelineArgs(
        phase_extractor=lambda x: "0",  # All images are from the same phase
        image_folder_name="Images",
        mask_folder_name=None,
        img_id_extractor=lambda x: 0,
        img_prefix="",
    )

    # Dictionary used to replace characters in file names to get numerical study_id

    def img_id_extractor(self, img_path: str) -> str:
        """Img id always 0 in this dataset."""
        return "0.png"

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        img_id = os.path.basename(img_path)
        img_basename = os.path.splitext(img_id)[0]
        # study id based on ids in source dataset, with replaced non-numerical characters
        for id in ids_dict.keys():
            img_basename = img_basename.replace(id, ids_dict[id])
        return img_basename

    def label_extractor(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        if "Y" in os.path.basename(img_path):
            return self.args["labels"]["Y"]
        elif "N" in os.path.basename(img_path) or "n" in os.path.basename(img_path):
            return self.args["labels"]["N"]
        else:
            return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.pipeline_args.label_extractor = lambda x: self.label_extractor(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
