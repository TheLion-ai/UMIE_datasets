"""Converts masks from xml files to png images with appropriate color encoding."""

import glob
import os
import plistlib
import re

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class CreateMasksFromXML(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_path: str,
        masks_path: str,
        dataset_uid: str,
        dataset_name: str,
        phases: dict,
        dataset_masks: list,
        target_colors: dict,
        zfill: int = 4,
        mask_folder_name: str = "Masks",
        **kwargs,
    ):
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_uid = dataset_uid
        self.dataset_name = dataset_name
        self.phases = phases
        self.dataset_masks = dataset_masks
        self.target_colors = target_colors
        self.zfill = zfill
        self.mask_folder_name = mask_folder_name

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):

        print("Creating masks from xml files...")
        mask_paths = glob.glob(f"{self.masks_path}/**/*.xml", recursive=True)
        for mask_path in tqdm(mask_paths):
            self.create_masks_from_xml(mask_path)
        return X

    def create_masks_from_xml(
        self,
        mask_path: str,
    ):

        with open(mask_path, mode="rb") as xml_file:
            segmentations = plistlib.load(xml_file)["Images"]

        study_id = os.path.basename(mask_path).split(".")[0]

        pattern = r"[-+]?\d*\.\d+|\d+"
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
                    if len(self.target_colors.keys()) <= 1:
                        color = list(self.target_colors.values())[0]
                    cv2.fillPoly(img, [np.array(points)], (color))

            for phase in self.phases.keys():
                if len(self.phases.keys()) <= 1:
                    filename_prefix = f"{self.dataset_uid}_{study_id}"
                else:
                    filename_prefix = f"{self.dataset_uid}_{study_id}_{phase}"

                new_path = os.path.join(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    self.mask_folder_name,
                    f"{filename_prefix}_{str(img_id).zfill(self.zfill)}.png",
                )
                cv2.imwrite(new_path, img)
