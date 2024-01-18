"""Library containing functions for testing pipelines."""
import glob
import os
from pathlib import Path

import cv2
import numpy as np


class DatasetTestingLibrary:
    """Library containing functions for testing pipelines."""

    @staticmethod
    def verify_file_tree(expected_file_tree: list, current_file_tree: list) -> bool:
        """Check if current directories and files are the same as expected."""
        # Change expected_file_tree folder name from expected_output to output
        expected_file_tree = [filepath.replace("expected_output", "output") for filepath in expected_file_tree]
        # Make sure lists have the same order
        expected_file_tree.sort()
        current_file_tree.sort()
        print(expected_file_tree)
        print(current_file_tree)

        return True if expected_file_tree == current_file_tree else False

    @staticmethod
    def verify_all_images_identical(expected_file_tree: list, current_file_tree: list) -> bool:
        """Check if images in current_file_tree are same as images in expected_file_tree."""
        expected_file_tree.sort()
        current_file_tree.sort()

        if len(expected_file_tree) != len(current_file_tree):
            return False

        for i in range(len(expected_file_tree)):
            expected_image_path = expected_file_tree[i]
            current_image_path = current_file_tree[i]

            expected_image = cv2.imread(expected_image_path)
            current_image = cv2.imread(current_image_path)
            print(expected_image_path)

            diff = cv2.subtract(expected_image, current_image)

            if np.sum(diff) != 0:
                return False

        return True

    @staticmethod
    def clean_up(directory_to_erase: Path) -> None:
        """Recursively remove files from directory_to_erase."""
        files_to_erase = glob.glob(f"{str(directory_to_erase)}/**", recursive=True)
        dirs_to_erase = []

        # Remove files first
        for file in files_to_erase:
            if not os.path.isdir(file):
                os.remove(file)
            else:
                dirs_to_erase.append(os.path.join(file))
        # Sort longest to shortest to delete child directories first
        dirs_to_erase = sorted(dirs_to_erase, key=len, reverse=True)

        # Remove directories
        for directory in dirs_to_erase:
            os.rmdir(directory)
