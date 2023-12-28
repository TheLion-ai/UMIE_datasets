"""Converts masks from xml files to png images with appropriate color encoding."""
import glob
import os
import plistlib
import re
from typing import Callable

import cv2
import numpy as np
import xmltodict
from tqdm import tqdm

from src.pipelines.base_pipeline import XmlKeys
from src.steps.base_step import BaseStep


class CreateMasksFromXML(BaseStep):
    """Converts masks from xml files to png images with appropriate color encoding."""

    def __init__(
        self,
        target_path: str,
        masks_path: str,
        dataset_uid: str,
        dataset_name: str,
        phases: dict,
        dataset_masks: list,
        mask_colors_source2target: dict,
        xml_keys: XmlKeys,
        zfill: int = 4,
        mask_folder_name: str = "Masks",
        study_id_extractor: Callable = lambda x: x,
        **kwargs: dict,
    ):
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            target_path (str): Path to the target folder.
            masks_path (str): Path to the folder with masks.
            dataset_uid (str): Unique identifier of the dataset.
            dataset_name (str): Name of the dataset.
            phases (dict): Dictionary with phases and their names.
            dataset_masks (list): List of masks to create.
            mask_colors_source2target(dict): Dictionary with target colors.
            xml_keys (XmlKeys): Arguments for xml parsing.
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 4.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
        """
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_uid = dataset_uid
        self.dataset_name = dataset_name
        self.phases = phases
        self.dataset_masks = dataset_masks
        self.mask_colors_source2target = mask_colors_source2target
        self.xml_keys = xml_keys
        self.zfill = zfill
        self.mask_folder_name = mask_folder_name
        self.study_id_extractor = study_id_extractor

    def transform(self, X: list) -> list:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        print("Creating masks from xml files...")
        mask_paths = glob.glob(os.path.join(self.masks_path, "**/*.xml"), recursive=True)
        # TODO Print warning if no xml paths available in path / path empty.
        # Make prompt whether to use source_path instead, otherwise skip this step,
        # or whole pipeline. If we skip step, all converted images will be wiped.
        for mask_path in tqdm(mask_paths):
            self.create_masks_from_xml(mask_path)
        return X

    def create_masks_from_xml(
        self,
        mask_path: str,
    ) -> None:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            mask_path (str): Path to the mask.
        """
        with open(mask_path, mode="rb") as xml_file:
            xml_content = xmltodict.parse(xml_file.read())
            if self.xml_keys.study not in xml_content.keys():
                return
            study = xml_content[self.xml_keys.study]
        study_id = self.study_id_extractor(mask_path)

        # Get annotators
        annotators = [study[self.xml_keys.main_file]] if not self.xml_keys.annotator else study[self.xml_keys.annotator]
        annotators = [annotators] if type(annotators) is not list else annotators
        for annotator in annotators:
            if self.xml_keys.obj not in annotator.keys():
                continue
            objs = [annotator] if not self.xml_keys.obj else annotator[self.xml_keys.obj]
            objs = [objs] if type(objs) is not list else objs
            for obj in objs:
                slices = [obj] if not self.xml_keys.slice else obj[self.xml_keys.slice]
                slices = [slices] if type(slices) is not list else slices
                for slice in slices:
                    self.process_slice(slice, study_id)

    def process_slice(self, slice: dict, study_id: str) -> None:
        """Process slice.

        Args:
            slice (dict): Slice.
            study_id (str): Study id.
        """
        rois = [slice] if not self.xml_keys.roi else slice[self.xml_keys.roi]
        rois = [rois] if type(rois) is not list else rois
        for roi in rois:
            if self.xml_keys.inclusion:
                if roi[self.xml_keys.inclusion] == "FALSE":
                    continue
            slice_id = (
                slice[self.xml_keys.slice_id] if self.xml_keys.slice_id in slice.keys() else roi[self.xml_keys.slice_id]
            )
            for phase_id in self.phases.keys():
                mask_path = self.create_file_path(slice_id, study_id, phase_id, mask=True)
                mask = self.add_roi_to_mask(roi, mask_path)
                cv2.imwrite(mask_path, mask)

    def add_roi_to_mask(self, roi: dict, mask_path: str) -> np.ndarray:
        """Add ROI to the mask.

        Args:
            roi (dict): ROI.
            mask_path (str): Path to the mask.
        Returns:
            np.ndarray: Mask.
        """
        mask = self.get_mask(mask_path)
        points = self.extract_points(roi)
        color = list(self.mask_colors_source2target.values())[0]
        if len(points) > 3:
            updated_mask = cv2.fillPoly(mask, [np.array(points)], (color))
            return updated_mask
        else:
            return mask

    def get_mask(self, mask_path: str) -> np.ndarray:
        """Create an empty mask or load an existing one.

        Args:
            mask_path (str): Path to the mask.
        Returns:
            np.ndarray: Mask.
        """
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((512, 512), np.uint8)
        return mask

    def extract_points(self, roi: dict) -> list:
        """Extract points from the ROI.

        Args:
            roi (dict): ROI.
        Returns:
            list: List of points.
        """
        extracted_points = []
        points = [roi[self.xml_keys.point]] if not self.xml_keys.point else roi[self.xml_keys.point]
        points = [points] if type(points) is not list else points
        for point in points:
            x, y = self.extract_coordinates(point)
            extracted_points.append([x, y])

        return extracted_points

    def extract_coordinates(self, point: str) -> tuple:
        """Extract coordinates from the point.

        Args:
            point (dict): Point.
        Returns:
            tuple: Coordinates.
        """
        if self.xml_keys.x and self.xml_keys.y:
            x, y = int(float(point[self.xml_keys.x])), int(float(point[self.xml_keys.y]))
        else:
            # If x and y are not specified, assume point is a string with coordinates
            # regex for extracting coordinates from point string
            pattern = r"[-+]?\d*\.\d+|\d+"
            x, y = re.findall(pattern, point)
            if x is not None and y is not None:
                # TODO: might crash for Nones
                x, y = int(x), int(y)
        return x, y


###############
# Stanford
#  with open(mask_path, mode="rb") as xml_file:
#     segmentations = plistlib.load(xml_file)["Images"]

# study_id = os.path.basename(mask_path).split(".")[0]
# print(study_id)
# pattern = r"[-+]?\d*\.\d+|\d+"
# for segmentation in segmentations:
#     img_id = segmentation["ImageIndex"]
#     # Create a blank mask
#     img = np.zeros((512, 512), np.uint8)
#     for roi in segmentation["ROIs"]:
#         if roi["NumberOfPoints"] > 0:
#             points = []
#             for point in roi["Point_px"]:
#                 x, y = re.findall(pattern, point)
#                 x, y = float(x), float(y)
#                 roi["NumberOfPoints"] > 0x, y = int(x), int(y)
#                 points.append([x, y])
#             if len(self.mask_colors_source2target.keys()) <= 1:
#                 color = list(self.mask_colors_source2target.values())[0]
#             cv2.fillPoly(img, [np.array(points)], (color))
