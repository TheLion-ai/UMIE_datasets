"""Recolors masks from default color to the color specified in the config."""

import glob
import os

import cv2
import nibabel as nib
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class RecolorMasks(BaseStep):
    """Recolors masks from default color to the color specified in the config."""

    def transform(self, X: list) -> list:
        """Recolors masks from default color to the color specified in the config.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        # Robust to multiple modalities and lack of masks for some images
        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        extension = "nii.gz" if self.output_mode == OutputMode.VOLUMES_3D else "png"
        mask_paths = glob.glob(os.path.join(root_path, f"**/{self.mask_folder_name}/*.{extension}"), recursive=True)
        print("Recoloring masks...")
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                if self.output_mode == OutputMode.VOLUMES_3D:
                    self.recolor_masks_3d(mask_path)
                else:
                    self.recolor_masks(mask_path)
        return X

    def recolor_masks_3d(self, mask_path: str) -> None:
        """Remap voxel values of a full NIfTI mask volume to the configured target colors.

        Reads from the original data and writes into a copy so each source color is remapped
        exactly once (no need for the 2D +255 guard); affine and header are preserved.

        Args:
            mask_path (str): Path to the ``.nii.gz`` mask volume.
        """
        nii = nib.load(mask_path)
        data = nii.get_fdata()  # type: ignore[attr-defined]
        recolored = data.copy()
        for mask_color in self.masks.values():
            recolored[data == mask_color["source_color"]] = mask_color["target_color"]
        recolored = recolored.astype(np.uint8)
        new_header = nii.header.copy()  # type: ignore[attr-defined]
        new_header.set_data_dtype(recolored.dtype)  # type: ignore[attr-defined]  # nibabel stub gap
        out = nib.Nifti1Image(recolored, affine=nii.affine, header=new_header)  # type: ignore[no-untyped-call,attr-defined]
        nib.save(out, mask_path)

    def recolor_masks(self, mask_path: str) -> None:
        """Recolors masks from default color to the color specified in the config.

        Args:
            mask_path (str): Path to the mask.
        """
        mask = cv2.imread(mask_path)
        # changing pixel values
        # add and then subtract factor of 255 to prevent changing color of the same area more than once
        mask = mask.astype("uint16")
        for mask_color in self.masks.values():
            np.place(mask, mask == mask_color["source_color"], mask_color["target_color"] + 255)

        mask[mask > 255] -= 255  # type: ignore[misc]  # numpy stub false positive on in-place subtract
        mask = mask.astype("uint8")
        cv2.imwrite(mask_path, mask)
