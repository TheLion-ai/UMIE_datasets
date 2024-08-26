"""Preprocessing pipeline for Brain Tumor Detection dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from umie_datasets.base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from umie_datasets.base.pipeline import BasePipeline, PipelineArgs
from umie_datasets.config.dataset_config import DatasetArgs, brain_tumor_detection
from umie_datasets.steps import (
    AddLabels,
    AddUmieIds,
    ConvertJpg2Png,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Detection dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return "0.png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Detection dataset."""

    # Dictionary used to replace characters in file names to get numerical study_id
    ids_dict = {" ": "0", "n": "1", "o": "2", "N": "3", "Y": "4"}

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        img_id = os.path.basename(img_path)
        img_basename = os.path.splitext(img_id)[0]
        # study id based on ids in source dataset, with replaced non-numerical characters
        for id in self.ids_dict.keys():
            img_basename = img_basename.replace(id, self.ids_dict[id])
        return img_basename


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Brain Tumor Detection dataset."""

    def _extract(self, img_path: str, *args: Any) -> list:
        """Extract label from img path."""
        if "Y" in os.path.basename(img_path):
            return self.labels["Y"]
        elif "N" in os.path.basename(img_path) or "n" in os.path.basename(img_path):
            return self.labels["N"]
        else:
            return []


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
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_detection)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name="Images",
            img_prefix="",
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"])
