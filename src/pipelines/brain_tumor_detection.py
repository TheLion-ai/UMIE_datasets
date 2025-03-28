"""Preprocessing pipeline for Brain Tumor Detection dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseLabelExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, brain_tumor_detection
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
    """Extractor for image IDs specific to the Brain Tumor Detection dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._return_zero()


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Detection dataset."""

    # Dictionary used to replace characters in file names to get numerical study_id
    ids_dict = {" ": "0", "n": "1", "o": "2", "N": "3", "Y": "4"}

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        img_basename = self._extract_filename(img_path)
        # study id based on ids in source dataset, with replaced non-numerical characters
        for id in self.ids_dict.keys():
            img_basename = img_basename.replace(id, self.ids_dict[id])
        return img_basename


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the Brain Tumor Detection dataset."""

    def _extract(self, img_path: str, *args: Any) -> tuple[list, list]:
        """Extract label from img path."""
        if "Y" in os.path.basename(img_path):
            return self.labels["Y"], ["tumor"]
        elif "N" in os.path.basename(img_path) or "n" in os.path.basename(img_path):
            return self.labels["N"], ["normal"]
        else:
            return [], []


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Brain Tumor Detection dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Brain Tumor Detection dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


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
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_detection)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            image_folder_name="Images",
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
