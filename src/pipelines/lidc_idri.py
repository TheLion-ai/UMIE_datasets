"""Preprocessing pipeline for LIDC-IDRI dataset."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import bs4 as bs
import cv2
import numpy as np

from base.creators.xml_mask import BaseXmlMaskCreator
from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.step import BaseStep
from config.dataset_config import DatasetArgs, lidc_idri
from constants import IMG_FOLDER_NAME
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CreateBlankMasks,
    CreateFileToDcmAttributeMapping,
    CreateFileTree,
    CreateMasksFromXml,
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

        return img_id


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the LIDC-IDRI dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        study_id = os.path.dirname(os.path.dirname(os.path.dirname(img_path))).split("-")[-1]

        return study_id


class XmlMaskCreator(BaseXmlMaskCreator):
    """Creator of masks based on xml files specific to the LIDC-IDRI dataset."""

    def _create(self, mask_path: str, caller: CreateMasksFromXml) -> None:
        """Create masks from xml file."""
        if caller.file_to_dicom_attr_mapping is None:
            raise FileNotFoundError(
                "file_to_dcm_attribute_mapping.json file not found. Dcm attribute mapping step was performed incorrectly."
            )

        file = open(mask_path, "r")
        contents = file.read()

        tree = bs.BeautifulSoup(contents, "xml")
        lesions = tree.find_all("nonNodule")

        for lesion in lesions:
            rois = lesion.find_all("roi")
            for roi in rois:
                image_sop_uids = roi.find_all("imageSOP_UID")
                if len(image_sop_uids) > 1:
                    print(f"More than one image_sop_uid for one lesion in file {mask_path}.")
                image_sop_uid = image_sop_uids[0].get_text()
                try:
                    corresponding_mask_path = caller.get_umie_mask_path(
                        caller.file_to_dicom_attr_mapping[image_sop_uid]
                    )
                except KeyError:
                    continue
                if os.path.exists(corresponding_mask_path):
                    current_mask = cv2.imread(corresponding_mask_path, cv2.IMREAD_GRAYSCALE)
                    points = list()
                    for locus in roi.find_all("locus"):
                        points.append([int(locus.find("xCoord").get_text()), int(locus.find("yCoord").get_text())])

                    color = caller.masks["Lesion"]["target_color"]
                    cv2.fillPoly(current_mask, [np.array(points)], (color))
                    print(f"Drawing on {corresponding_mask_path}")
                    cv2.imwrite(corresponding_mask_path, current_mask)

        nodules = tree.find_all("unblindedReadNodule")
        for nodule in nodules:
            rois = nodule.find_all("roi")
            for roi in rois:
                image_sop_uids = roi.find_all("imageSOP_UID")
                if len(image_sop_uids) > 1:
                    print(f"More than one image_sop_uid for one nodule in file {mask_path}.")
                image_sop_uid = image_sop_uids[0].get_text()
                try:
                    corresponding_mask_path = caller.get_umie_mask_path(
                        caller.file_to_dicom_attr_mapping[image_sop_uid]
                    )
                except KeyError:
                    continue

                if os.path.exists(corresponding_mask_path):
                    current_mask = cv2.imread(corresponding_mask_path, cv2.IMREAD_GRAYSCALE)
                    points = list()
                    for edge_map in roi.find_all("edgeMap"):
                        points.append(
                            [int(edge_map.find("xCoord").get_text()), int(edge_map.find("yCoord").get_text())]
                        )

                    inclusion_string = roi.find("inclusion").get_text()
                    if inclusion_string == "TRUE":
                        color = caller.masks["Nodule"]["target_color"]
                    elif inclusion_string == "FALSE":
                        color = 0
                    else:
                        print(f"Unsupported inclusion value {inclusion_string}")
                        color = 0
                    cv2.fillPoly(current_mask, [np.array(points)], (color))
                    cv2.imwrite(corresponding_mask_path, current_mask)


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
        ("create_masks_from_xml", CreateMasksFromXml),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: lidc_idri)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            xml_mask_creator=XmlMaskCreator(),
            dicom_mapping_attribute="SOPInstanceUID",
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
