"""Preprocessing pipeline for brain tumor classification dataset."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
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
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Classification dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._return_zero()


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Classification dataset."""

    """
    Img names are not unique. To make them unique there is system based on image folder path
    /Training/ -> 0 added /Testing/ -> 1 added then there are 4 folders one for each label.
    This fact is used to assign unique ids
    """
    unique_id_conversion_dict = {
        "glioma_tumor": "00",
        "meningioma_tumor": "01",
        "pituitary_tumor": "10",
        "no_tumor": "11",
    }

    def _extract(self, img_path: str) -> str:
        unique_id = ""
        if "Training" in img_path:
            unique_id = "0"
        else:
            unique_id = "1"

        parent_directory = self._extract_parent_dir(img_path, parent_dir_level=1)

        image_folder = self._extract_filename(parent_directory)
        unique_id = unique_id + self.unique_id_conversion_dict[image_folder]

        # after conversion to png there are additional png files
        jpg_files = [
            Path(file_path).stem
            for file_path in Path(parent_directory).iterdir()
            if file_path.name.lower().endswith(".jpg")
        ]
        # makes sure that we get the same order
        jpg_files.sort()

        img_basename = self._extract_filename(img_path)

        return unique_id + str(jpg_files.index(img_basename))


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Brain Tumor Classification dataset."""

    def _extract(self, source_img_path: str, *args: Any) -> tuple[list, list]:
        image_folder = os.path.basename(os.path.dirname(source_img_path))
        source_label = image_folder
        return self.labels[image_folder], [source_label]


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Brain Tumor Classification dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Brain Tumor Classification dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


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
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_classification)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name="Images",
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            img_prefix="",
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.label_extractor = LabelExtractor(self.args["labels"])

        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
