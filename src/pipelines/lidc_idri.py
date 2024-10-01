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
from config.dataset_config import DatasetArgs, lidc_idri
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

    def get_mask_path(self, roi: bs.BeautifulSoup, caller: CreateMasksFromXml) -> str | None:
        """Get mask path from mapped dicom 'image_sop_uid' parameter."""
        if caller.file_to_dicom_attr_mapping is None:
            raise FileNotFoundError(
                "file_to_dcm_attribute_mapping.json file not found. Dcm attribute mapping step was performed incorrectly."
            )

        image_sop_uids = roi.find_all("imageSOP_UID")
        if len(image_sop_uids) > 1:
            print("More than one image_sop_uid for one item.")
        image_sop_uid = image_sop_uids[0].get_text()
        try:
            corresponding_mask_path = caller.get_umie_mask_path(caller.file_to_dicom_attr_mapping[image_sop_uid])
            return corresponding_mask_path
        except KeyError:
            return None

    def draw_on_mask(self, path: str, color: tuple, points: list) -> None:
        """Draw polygon from points on mask from path in selected color."""
        current_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv2.fillPoly(current_mask, [np.array(points)], (color))
        cv2.imwrite(path, current_mask)

    def _create(self, mask_path: str, caller: CreateMasksFromXml) -> None:
        """Create masks from xml file."""
        file = open(mask_path, "r")
        contents = file.read()

        # Draw lesions from xml file
        tree = bs.BeautifulSoup(contents, "xml")
        lesions = tree.find_all("nonNodule")

        for lesion in lesions:
            rois = lesion.find_all("roi")
            for roi in rois:
                corresponding_mask_path = self.get_mask_path(roi, caller)
                if corresponding_mask_path is None:
                    continue

                points = list()
                for locus in roi.find_all("locus"):
                    points.append([int(locus.find("xCoord").get_text()), int(locus.find("yCoord").get_text())])

                if os.path.exists(corresponding_mask_path):
                    color = caller.masks["Lesion"]["target_color"]
                    self.draw_on_mask(corresponding_mask_path, color, points)

        # Draw nodules from xml file
        nodules = tree.find_all("unblindedReadNodule")
        for nodule in nodules:
            rois = nodule.find_all("roi")
            for roi in rois:
                corresponding_mask_path = self.get_mask_path(roi, caller)
                if corresponding_mask_path is None:
                    continue

                if os.path.exists(corresponding_mask_path):
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

                    self.draw_on_mask(corresponding_mask_path, color, points)


@dataclass
class LidcIdriPipeline(BasePipeline):
    """Preprocessing pipeline for LIDC-IDRI dataset."""

    name: str = "lidc_idri"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("store_source_paths", StoreSourcePaths),
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
