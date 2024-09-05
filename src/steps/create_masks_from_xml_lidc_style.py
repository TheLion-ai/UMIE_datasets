"""Converts masks from xml files to png images with appropriate color encoding."""

import glob
import json
import os
import re

import bs4 as bs
import cv2
import numpy as np
from tqdm import tqdm

from base.step import BaseStep


class CreateMasksFromXmlLidcStyle(BaseStep):
    """Converts masks from xml files to png images with appropriate color encoding."""

    def transform(self, X: list) -> list:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        print("Creating masks from xml files...")

        file_to_dicom_attr_mapping_file = os.path.join(self.target_path, "file_to_dcm_attribute_mapping.json")
        if os.path.exists(file_to_dicom_attr_mapping_file):
            self.file_to_dicom_attr_mapping = json.load(open(file_to_dicom_attr_mapping_file))
        else:
            raise FileNotFoundError(
                "file_to_dcm_attribute_mapping.json file not found. Dcm attribute mapping step was performed incorrectly."
            )

        mask_xml_paths = glob.glob(f"{self.masks_path}/**/*.xml", recursive=True)
        # TODO Print warning if no xml paths available in path / path empty.
        # Make prompt whether to use source_path instead, otherwise skip this step,
        # or whole pipeline. If we skip step, all converted images will be wiped.
        for mask_xml_path in tqdm(mask_xml_paths):
            self.create_masks_from_xml_lidc_style(mask_xml_path)
            # break
        return X

    def create_masks_from_xml_lidc_style(
        self,
        mask_path: str,
    ) -> None:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            mask_path (str): Path to the mask.
        """
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
                    corresponding_mask_path = self.get_umie_mask_path(self.file_to_dicom_attr_mapping[image_sop_uid])
                except KeyError:
                    continue
                if os.path.exists(corresponding_mask_path):
                    current_mask = cv2.imread(corresponding_mask_path, cv2.IMREAD_GRAYSCALE)
                    points = list()
                    for locus in roi.find_all("locus"):
                        points.append([int(locus.find("xCoord").get_text()), int(locus.find("yCoord").get_text())])

                    color = self.masks["Lesion"]["target_color"]
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
                    corresponding_mask_path = self.get_umie_mask_path(self.file_to_dicom_attr_mapping[image_sop_uid])
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
                        color = self.masks["Nodule"]["target_color"]
                    elif inclusion_string == "FALSE":
                        color = 0
                    else:
                        print(f"Unsupported inclusion value {inclusion_string}")
                        color = 0
                    cv2.fillPoly(current_mask, [np.array(points)], (color))
                    cv2.imwrite(corresponding_mask_path, current_mask)
