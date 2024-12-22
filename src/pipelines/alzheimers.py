"""Preprocessing pipeline for Alzheimers dataset."""
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, alzheimers
from constants import IMG_FOLDER_NAME
from steps import (
    AddLabels,
    AddUmieIds,
    ConvertJpg2Png,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Alzheimer's dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if "train" in img_path:
            return self._extract_first_two_chars(img_path, True)
        else:
            return self._extract_first_two_chars(img_path)


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
        basename = self._extract_filename(img_path)
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
            folder = self._extract_filename(self._extract_parent_dir(img_path, parent_dir_level=-1))
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


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Alzheimer's dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Alzheimer's dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


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
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: alzheimers)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name=IMG_FOLDER_NAME,
            img_prefix="",
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["label_extractor"] = LabelExtractor(self.args["labels"])
