"""Preprocessing pipeline for Brain Tumor Detection dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class BrainTumorDetectionPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Detection dataset."""

    name: str = field(default="Brain_Tumor_Detection")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("convert_jpg2png", ConvertJpg2Png),
            ("add_new_ids", AddNewIds),
            ("add_new_ids", AddLabels),
            ("delete_temp_files", DeleteTempFiles),
            ("delete_temp_png", DeleteTempPng),
        ]
    )
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name="Images",
            mask_folder_name=None,
            img_id_extractor=lambda x: 0,
            img_prefix="",
        )
    )

    # Dictionary used to replace characters in file names to get numerical study_id
    ids_dict = {" ": "0", "n": "1", "o": "2", "N": "3", "Y": "4"}

    def img_id_extractor(self, img_path: str) -> str:
        """Img id always 0 in this dataset."""
        return "0"

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        img_id = os.path.basename(img_path)
        img_basename = os.path.splitext(img_id)[0]
        # study id based on ids in source dataset, with replaced non-numerical characters
        for id in self.ids_dict.keys():
            img_basename = img_basename.replace(id, self.ids_dict[id])
        return img_basename

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        if "Y" in os.path.basename(img_path):
            return ["Tumor"]
        elif "N" in os.path.basename(img_path) or "n" in os.path.basename(img_path):
            return ["good"]
        else:
            return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.pipeline_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
