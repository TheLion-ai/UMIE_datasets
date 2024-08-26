"""Delete images without masks."""

import glob
import os

import cv2
import jsonlines
import numpy as np
from tqdm import tqdm

from umie_datasets.base.step import BaseStep


class DeleteImgsWithNoAnnotations(BaseStep):
    """Delete images without masks."""

    def transform(self, X: str) -> list:
        """Delete images without annotations.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        root_path = os.path.dirname(os.path.dirname(X[0]))

        self.json_lines = {}
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                umie_file_name = os.path.basename(obj["umie_path"])
                self.json_lines[umie_file_name] = obj

        if self.mask_folder_name:
            mask_paths = glob.glob(f"{os.path.join(root_path, self.mask_folder_name)}/**/*.png", recursive=True)
            self.mask_names = [os.path.basename(mask) for mask in mask_paths]
        print("Deleting images without annotations...")

        for img_path in tqdm(X):
            self.delete_imgs_with_no_annotations(img_path, root_path)

        # Create new list of paths after the deletion
        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)

        with jsonlines.open(self.json_path, mode="w") as writer:
            remaining_files = set(os.path.basename(path) for path in new_paths)
            for k, obj in self.json_lines.items():
                if k in remaining_files:
                    writer.write(obj)

        return new_paths

    def delete_imgs_with_no_annotations(self, img_path: str, root_path: str) -> None:
        """Delete images without any annotations i.e. labels or masks.

        Args:
            img_path (str): Path to the image.
            root_path (str): Path to the root folder.
        """
        img_name = os.path.basename(img_path)
        no_label = False
        if self.json_lines[img_name]["labels"] == []:
            no_label = True
        if self.masks_path:
            mask_path = os.path.join(root_path, self.mask_folder_name, img_name)
            if img_name not in self.mask_names:
                if no_label:
                    # If there is no mask and no label
                    os.remove(img_path)
            # If there is a mask but it is empty
            elif np.unique(cv2.imread(mask_path)).shape[0] == 1:
                if no_label:
                    # If there is a blank mask and no label
                    os.remove(img_path)
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
