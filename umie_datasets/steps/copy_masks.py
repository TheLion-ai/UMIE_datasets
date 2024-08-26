"""Copy masks to a new folder structure."""

import glob
import os
import shutil

from umie_datasets.base.step import BaseStep


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
        mask_paths = glob.glob(os.path.join(self.masks_path, "**/*.png"), recursive=True)
        for mask_path in mask_paths:
            if (
                os.path.exists(mask_path)
                and self.segmentation_prefix is not None
                and self.segmentation_prefix in mask_path
            ):
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
        if self.segmentation_prefix not in img_path:
            return None
        if self.mask_selector is not None and self.mask_selector in img_id:
            img_id = img_id.replace(self.mask_selector, "")
        for phase_id in self.phases.keys():
            if self.multiple_masks_selector and any(
                original_mask_selector in img_path for original_mask_selector in self.multiple_masks_selector.keys()
            ):
                continue
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
