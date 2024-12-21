"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
import plistlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from base.creators.xml_mask import BaseXmlMaskCreator
from base.extractors import BaseImgIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, coca
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CreateFileTree,
    CreateMasksFromXml,
    DeleteImgsWithNoAnnotations,
    DeleteTempPng,
    GetFilePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Stanford COCA dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        # Image id is in the source file name after the last underscore
        return self._extract_by_separator(img_path, "-")


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Stanford COCA dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study name is the folder two levels above the image
        return self._extract_parent_dir(img_path, parent_dir_level=-2, include_path=False)


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Stanford COCA dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return path.endswith(".dcm")


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Stanford COCA dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return path.endswith(".xml")


class XmlMaskCreator(BaseXmlMaskCreator):
    """Creator of masks based on xml files specific to the Stanford COCA dataset."""

    def _create(self, mask_path: str, caller: CreateMasksFromXml) -> None:
        """Create masks from xml file."""
        with open(mask_path, mode="rb") as xml_file:
            segmentations = plistlib.load(xml_file)["Images"]

        study_id = os.path.basename(mask_path).split(".")[0]

        pattern = r"[-+]?\d*\.\d+|\d+"  # Extract numbers from string
        for segmentation in segmentations:
            img_id = segmentation["ImageIndex"]
            img = np.zeros((512, 512), np.uint8)
            for roi in segmentation["ROIs"]:
                if roi["NumberOfPoints"] > 0:
                    points = []
                    for point in roi["Point_px"]:
                        x, y = re.findall(pattern, point)
                        x, y = float(x), float(y)
                        x, y = int(x), int(y)
                        points.append([x, y])
                    # TODO: add case when there is more than one color
                    color = list(caller.masks.values())[0]["target_color"]
                    cv2.fillPoly(img, [np.array(points)], (color))

            for phase_id, phase_name in caller.phases.items():
                filename_prefix = f"{caller.dataset_uid}_{phase_id}_{study_id}"

                new_path = os.path.join(
                    caller.target_path,
                    f"{caller.dataset_uid}_{caller.dataset_name}",
                    phase_name,
                    caller.mask_folder_name,
                    f"{filename_prefix}_{str(img_id).zfill(caller.zfill)}.png",
                )
                cv2.imwrite(new_path, img)


@dataclass
class COCAPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford COCA dataset."""

    name: str = "coca"  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("convert_dcm2png", ConvertDcm2Png),
        ("create_masks_from_xml", CreateMasksFromXml),
        ("add_new_ids", AddUmieIds),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks),
        ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: coca)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            # Image id is in the source file name after the last underscore
            img_id_extractor=ImgIdExtractor(),
            # Study name is the folder two levels above the image
            study_id_extractor=StudyIdExtractor(),
            xml_mask_creator=XmlMaskCreator(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
