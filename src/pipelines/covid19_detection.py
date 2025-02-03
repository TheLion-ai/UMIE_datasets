"""Preprocessing pipeline for Covid 19 detection dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, covid19_detection
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
    """Extractor for image IDs specific to the Covid 19 detection dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._return_zero()


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Covid 19 detection dataset."""

    # Images in the dataset has non-unique ids between classes and folders.
    # Dictionaries below are used to make them unique across whole dataset.

    # Ids added to image names in ValData folder based on their classes
    ids_dict_val = {
        "Normal": "000",
        "BacterialPneumonia": "111",
        "ViralPneumonia": "222",
        "COVID-19": "333",
    }
    # Ids added to image names in NonAugmentedTrain folder based on their classes
    ids_dict_non_aug = {
        "Normal": "444",
        "BacterialPneumonia": "555",
        "ViralPneumonia": "666",
        "COVID-19": "777",
    }

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        """Get study id with added postfix depending on source location to prevent repeated names."""
        img_basename = self._extract_filename(img_path)
        parent_basename = self._extract_parent_dir(img_path, parent_dir_level=-1, include_path=False)
        if "ValData" in img_path:
            study_id = img_basename + self.ids_dict_val[parent_basename]
        elif "NonAugmentedTrain" in img_path:
            study_id = img_basename + self.ids_dict_non_aug[parent_basename]
        else:
            study_id = img_basename
        study_id = study_id.replace("_", "")

        return study_id


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Covid 19 detection dataset."""

    def _extract(self, img_path: str, *args: Any) -> tuple[list, list]:
        """Extract label from img path."""
        label = os.path.basename(os.path.dirname(img_path))
        return self.labels[label], [label]


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Covid 19 detection dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Covid 19 detection dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


@dataclass
class COVID19DetectionPipeline(BasePipeline):
    """Preprocessing pipeline for Covid 19 detection dataset."""

    name: str = "covid19_detection"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("get_source_paths", StoreSourcePaths),
        ("convert_jpg2png", ConvertJpg2Png),
        ("add_new_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: covid19_detection)
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
