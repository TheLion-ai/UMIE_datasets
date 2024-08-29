"""Preprocessing pipeline for LIDC-IDRI dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, lidc_idri
from constants import IMG_FOLDER_NAME
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CreateBlankMasks,
    CreateFileToDcmAttributeMapping,
    CreateFileTree,
    CreateMasksFromXmlLidcStyle,
    DeleteImgsWithNoAnnotations,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the LIDC-IDRI dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract image id from img path."""
        img_id = str(os.path.splitext(img_path)[0].split("-")[-1] + ".png")
        # print(f"Image: {img_id}")
        return img_id


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the LIDC-IDRI dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        study_id = os.path.dirname(os.path.dirname(os.path.dirname(img_path))).split("-")[-1]
        # print(f"Study: {study_id}")
        return study_id


@dataclass
class LidcIdriPipeline(BasePipeline):
    """Preprocessing pipeline for LIDC-IDRI dataset."""

    name: str = "lidc_idri"  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
        ("create_file_tree", CreateFileTree),
        ("create_file_to_dcm_attribute_mapping", CreateFileToDcmAttributeMapping),
        ("convert_dcm2png", ConvertDcm2Png),
        ("add_new_ids", AddUmieIds),
        ("create_blank_masks", CreateBlankMasks),
        ("create_masks_from_xml_lidc_style", CreateMasksFromXmlLidcStyle),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: lidc_idri)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            # image_folder_name=IMG_FOLDER_NAME,
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            dicom_mapping_attribute="SOPInstanceUID",
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
