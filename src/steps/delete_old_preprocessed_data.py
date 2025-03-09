"""Delete old preprocessed data."""
import os
import shutil

from base.step import BaseStep


class DeleteOldPreprocessedData(BaseStep):
    """Delete old preprocessed data."""

    def transform(self, X: list) -> list:
        """Delete old preprocessed data.

        Args:
            X (list): List of input data paths or pipeline inputs.
        Returns:
            X (list): List of inputs unchanged.
        """
        directory_to_delete = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")

        if os.path.exists(directory_to_delete):
            print(f"Deleting old preprocessed data from: {directory_to_delete}")
            shutil.rmtree(directory_to_delete)
        else:
            print(f"No old preprocessed data to delete in: {directory_to_delete}")

        return X
