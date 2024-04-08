"""Converts masks from xml files to png images with appropriate color encoding."""

import glob
import os
import plistlib
import re

import cv2
import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class CreateMasksFromXML(TransformerMixin):
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
        zfill: int = 4,
        mask_folder_name: str = "Masks",
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
            mask_colors_source2target (dict): Dictionary with target colors.
            zfill (int, optional): Number of zeros to fill the image id. Defaults to 4.
            mask_folder_name (str, optional): Name of the folder with masks. Defaults to "Masks".
        """
        self.target_path = target_path
        self.masks_path = masks_path
        self.dataset_uid = dataset_uid
        self.dataset_name = dataset_name
        self.phases = phases
        self.dataset_masks = dataset_masks
        self.mask_colors_source2target = mask_colors_source2target
        self.zfill = zfill
        self.mask_folder_name = mask_folder_name

    def transform(self, X: list) -> list:
        """Convert masks from xml files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        print("Creating masks from xml files...")
        mask_paths = glob.glob(f"{self.masks_path}/**/*.xml", recursive=True)
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
                    color = list(self.mask_colors_source2target.values())[0]
                    cv2.fillPoly(img, [np.array(points)], (color))

            for phase_id, phase_name in self.phases.items():
                filename_prefix = f"{self.dataset_uid}_{phase_id}_{study_id}"

                new_path = os.path.join(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase_name,
                    self.mask_folder_name,
                    f"{filename_prefix}_{str(img_id).zfill(self.zfill)}.png",
                )
                cv2.imwrite(new_path, img)
