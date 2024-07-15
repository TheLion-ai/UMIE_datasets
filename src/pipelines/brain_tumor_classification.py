"""Preprocessing pipeline for brain tumor classification dataset."""


import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, brain_tumor_classification
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


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Classification dataset."""

    def _extract(self, img_path: str) -> str:
        return "0.png"


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Classification dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path. Img names are not unique.

        To make them unique there is system based on image folder path
        /Training/ -> 0 added /Testing/ -> 1 added then there are 4 folders one for each label.
        This fact is used to assign unique ids
        """
        unique_id_conversion_dict = {
            "glioma_tumor": "00",
            "meningioma_tumor": "01",
            "pituitary_tumor": "10",
            "no_tumor": "11",
        }
        unique_id = ""
        if "Training" in img_path:
            unique_id = "0"
        else:
            unique_id = "1"

        image_folder = os.path.basename(os.path.dirname(img_path))
        unique_id = unique_id + unique_id_conversion_dict[image_folder]

        parent_directory = Path(img_path).parent
        files_in_parent_directory = os.listdir(parent_directory)

        # after conversion to png there are additional png files
        jpg_files = [Path(file).stem for file in files_in_parent_directory if file.lower().endswith(".jpg")]
        # makes sure that we get the same order
        jpg_files.sort()

        img_filename = Path(img_path).name
        return unique_id + str(jpg_files.index(Path(img_filename).stem))


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Brain Tumor Classification dataset."""

    def _extract(self, source_img_path: str, *args: Any) -> list:
        image_folder = os.path.basename(os.path.dirname(source_img_path))
        return self.labels[image_folder]


@dataclass
class BrainTumorClassificationPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Classification dataset."""

    name: str = "Brain_Tumor_Classification"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("convert_jpg2png", ConvertJpg2Png),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_png", DeleteTempPng),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_classification)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name="Images",
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            img_prefix="",
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.label_extractor = LabelExtractor(self.args["labels"])

        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
