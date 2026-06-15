"""Delete temporary PNG files created by other steps in the source directory. Use with caution some source data may already be in the PNG format."""

import glob
import os
from typing import Optional

from tqdm import tqdm

from base.step import BaseStep
from constants import OutputMode


class DeleteTempPng(BaseStep):
    """Delete temporary png files created by other steps in the source directory."""

    def fit(self, X: list, y: Optional[list] = None) -> "DeleteTempPng":
        """Fit method, no-op for sklearn pipeline compatibility; returns self."""
        # for sklearn compatibility
        return self

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Delete temporary PNG files created by other steps in the source directory, including nested directories.

        Args:
            X (list): List of paths to the target images.
        Returns:
            list: List of paths to the target images.
        """
        # In 3D mode no temporary PNGs are produced (volumes stay as .nii.gz), so this is a no-op.
        if self.output_mode == OutputMode.VOLUMES_3D:
            return X
        print("Removing temporary PNG files...")
        png_paths = glob.glob(os.path.join(self.source_path, "**", "*.png"), recursive=True)
        for png_path in tqdm(png_paths):
            self.remove_temp_png(png_path)
        return X

    def remove_temp_png(self, png_path: str) -> None:
        """Remove temporary PNG files created by other steps in the source directory.

        Args:
            png_path (str): Path to the PNG file.
        """
        os.remove(png_path)
