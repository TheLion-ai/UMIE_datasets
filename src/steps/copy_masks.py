"""Copy masks to a new folder structure."""

import os
import shutil
from typing import Callable

from base.step import BaseStep
from tqdm import tqdm
from base.extractors.img_id import BaseImgIdExtractor

class CopyMasks(BaseStep):
    """Copy masks to a new folder structure."""


    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Copy masks to a new folder structure.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Copying masks...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            path_el = img_path.rsplit(self.img_prefix, 1)
            mask_path = self.segmentation_prefix.join(path_el)
            if os.path.exists(mask_path):
                self.copy_masks(mask_path)
        return X

    #
    def copy_masks(self, img_path: str) -> None:
        """Copy PNG masks to a new folder structure.

        Args:
            img_path (str): Path to the image.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        # TODO: remove duplicate code from add_new_ids.py, Move this step to add_new_ids???
        if self.mask_selector in img_id:
            img_id = img_id.replace(self.mask_selector, "")
        if self.segmentation_prefix not in img_path:
            return None
        for phase_id in self.phases.keys():
            if phase_id == self.phase_extractor(img_path):
                phase_name = self.phases[phase_id]
                new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
                if "." not in new_file_name:
                    new_file_name = new_file_name + ".png"
                new_path = os.path.join(
                    self.target_path,
                    f"{self.dataset_uid}_{self.dataset_name}",
                    phase_name,
                    self.mask_folder_name,
                    new_file_name,
                )

                if not os.path.exists(new_path):
                    shutil.copy2(img_path, new_path)
