"""Preprocessing pipeline for Alzheimers dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from config import dataset_config
from src.constants import IMG_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class KneeOsteoarthritisPipeline(BasePipeline):
    """Preprocessing pipeline for Knee Osteoarthritis dataset."""

    name: str = field(default="Knee_Osteoarthritis")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("add_new_ids", AddNewIds),
            ("add_new_ids", AddLabels),
            ("delete_temp_files", DeleteTempFiles),
        ]
    )
    dataset_args: dataset_config.knee_osteoarthritis = field(default_factory=lambda: dataset_config.knee_osteoarthritis)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name=IMG_FOLDER_NAME,
            mask_folder_name=None,
        )
    )

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        # replace letters and delete underscore from filenames
        # letters can't be deleted because they make names unique
        study_id = os.path.splitext(os.path.basename(img_path))[0].replace("R", "0").replace("L", "1").replace("_", "")
        study_id = study_id + os.path.basename(os.path.dirname(img_path))
        return study_id

    def img_id_extractor(self, img_path: str) -> str:
        """Each study has only 1 image."""
        return "0"

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        label = os.path.basename(os.path.dirname(img_path))

        return self.args["labels"][label]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        self.pipeline_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
