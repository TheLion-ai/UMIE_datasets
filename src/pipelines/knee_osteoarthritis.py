"""Preprocessing pipeline for Alzheimers dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, knee_osteoarthritis
from constants import IMG_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    CreateFileTree,
    DeleteTempFiles,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Knee Osteoarthritis dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract image id from img path."""
        return "0.png"


@dataclass
class KneeOsteoarthritisPipeline(BasePipeline):
    """Preprocessing pipeline for Knee Osteoarthritis dataset."""

    name: str = "Knee_Osteoarthritis"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("get_source_paths", StoreSourcePaths),
        ("add_new_ids", AddUmieIds),
        ("add_new_ids", AddLabels),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: knee_osteoarthritis)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name=IMG_FOLDER_NAME,
            mask_folder_name=None,
            img_id_extractor=ImgIdExtractor(),
        )
    )

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        # replace letters and delete underscore from filenames
        # letters can't be deleted because they make names unique
        study_id = os.path.splitext(os.path.basename(img_path))[0].replace("R", "0").replace("L", "1").replace("_", "")
        study_id = study_id + os.path.basename(os.path.dirname(img_path))
        return study_id

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        label = os.path.basename(os.path.dirname(img_path))

        return self.args["labels"][label]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        self.pipeline_args.get_label = self.get_label
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
