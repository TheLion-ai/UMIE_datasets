"""Delete temporary files."""
import os

from base.step import BaseStep


class DeleteTempFiles(BaseStep):
    """Delete temporary files."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Delete temporary files.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        source_paths_file = os.path.join(self.target_path, "source_paths.json")
        if os.path.exists(source_paths_file):
            os.remove(source_paths_file)

        return X
