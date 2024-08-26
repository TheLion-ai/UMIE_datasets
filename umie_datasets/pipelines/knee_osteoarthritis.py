"""Preprocessing pipeline for Alzheimers dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from umie_datasets.base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from umie_datasets.base.pipeline import BasePipeline, PipelineArgs
from umie_datasets.config.dataset_config import DatasetArgs, knee_osteoarthritis
from umie_datasets.constants import IMG_FOLDER_NAME
from umie_datasets.steps import (
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


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Knee Osteoarthritis dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # replace letters and delete underscore from filenames
        # letters can't be deleted because they make names unique
        # replace letters and delete underscore from filenames
        # letters can't be deleted because they make names unique
        study_id = os.path.splitext(os.path.basename(img_path))[0].replace("R", "0").replace("L", "1").replace("_", "")
        study_id = study_id + os.path.basename(os.path.dirname(img_path))
        return study_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Knee Osteoarthritis dataset."""

    def _extract(self, img_path: str, *args: Any) -> list:
        """Extract label from img path."""
        source_label = os.path.basename(os.path.dirname(img_path))
        return self.labels[source_label]


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
            image_folder_name=IMG_FOLDER_NAME,
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"])
