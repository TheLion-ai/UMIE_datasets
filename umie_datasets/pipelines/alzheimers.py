"""Preprocessing pipeline for Alzheimers dataset."""
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from umie_datasets.base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from umie_datasets.base.pipeline import BasePipeline, PipelineArgs
from umie_datasets.config.dataset_config import DatasetArgs, alzheimers
from umie_datasets.constants import IMG_FOLDER_NAME
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
    """Extractor for image IDs specific to the Alzheimer's dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        img_id = os.path.basename(img_path)
        ext = os.path.splitext(img_id)[1]
        if "train" in img_path:
            img_id = "00"
        else:
            img_id = img_id[:2]
        img_id = img_id + ext
        return img_id


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Alzheimer's dataset."""

    # Filenames in source directory are not unique, so additional id is added to each study_id, based on parent folder
    # name to make them unique across the whole dataset.

    # Ids added to study id based on the name of parent folder name for a 'test' source directory.
    ids_dict_test = {
        "NonDemented": "0",
        "VeryMildDemented": "1",
        "MildDemented": "2",
        "ModerateDemented": "3",
    }
    # Ids added to study id based on the name of parent folder name for a 'train' source directory.
    ids_dict_train = {
        "nonDem": "000",
        "verymildDem": "111",
        "mildDem": "222",
        "moderateDem": "333",
    }

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # If brackets exist in filename, then study id is within them, else it is 0.
        if "(" in basename:
            pattern = r"[()]"
            study_id = re.split(pattern, basename)[1]
        else:
            study_id = "0"
        # Add identifier based on folder name to make new file name unique across dataset
        if "train" in img_path:
            for id in self.ids_dict_train.keys():
                basename = basename.replace(id, self.ids_dict_train[id])
            study_id = study_id + basename
        else:
            folder = os.path.basename(os.path.dirname(img_path))
            for id in self.ids_dict_test.keys():
                folder = folder.replace(id, self.ids_dict_test[id])
            study_id = folder + study_id
        return study_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Alzheimer's dataset."""

    def _extract(self, img_path: str, *args: Any) -> str:
        """Extract label from img path."""
        source_label = os.path.basename(os.path.dirname(img_path))
        radlex_label = self.labels[source_label]
        return radlex_label


@dataclass
class AlzheimersPipeline(BasePipeline):
    """Preprocessing pipeline for Alzheimers dataset."""

    name: str = "alzheimers"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("convert_jpg2png", ConvertJpg2Png),
        ("add_umie_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_png", DeleteTempPng),
        ("delete_temp_files", DeleteTempFiles),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: alzheimers)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name=IMG_FOLDER_NAME,
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
