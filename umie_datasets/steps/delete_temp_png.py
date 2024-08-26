"""Delete temporary PNG files created by other steps in the source directory. Use with caution some source data may already be in the PNG format."""

import glob
import os

from tqdm import tqdm

from umie_datasets.base.step import BaseStep


class DeleteTempPng(BaseStep):
    """Delete temporary png files created by other steps in the source directory."""

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
