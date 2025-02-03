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
        # Find differences
        expected_set = set(expected_file_tree)
        current_set = set(current_file_tree)

        differences = expected_set.symmetric_difference(current_set)
        if differences:
            expected_differences = expected_set - current_set
            if expected_differences:
                print(f"Expected but not in current: {expected_differences}\n")
            else:
                current_differences = current_set - expected_set
                print(f"Found in current but not expected: {current_differences}")
            return False
        return True

    @staticmethod
    def verify_all_images_identical(expected_file_tree: list, current_file_tree: list) -> bool:
        """Check if images in current_file_tree are same as images in expected_file_tree."""
        expected_file_tree.sort()
        current_file_tree.sort()

        if len(expected_file_tree) != len(current_file_tree):
            print(f"Mismatch in number of files: {len(expected_file_tree)} expected vs {len(current_file_tree)} found.")
            return False

        all_identical = True

        for i in range(len(expected_file_tree)):
            expected_image_path = expected_file_tree[i]
            current_image_path = current_file_tree[i]

            expected_image = cv2.imread(expected_image_path)
            current_image = cv2.imread(current_image_path)
            if expected_image.shape != current_image.shape:
                print(f"These images are not identical: {expected_image_path}, {current_image_path}")
                all_identical = False

            diff = cv2.subtract(expected_image, current_image)
            if np.any(diff):
                mismatch_coords = np.column_stack(np.where(diff != 0))
                first_mismatch = mismatch_coords[0] if len(mismatch_coords) > 0 else None
                print(f"These images are not identical: {expected_image_path}, {current_image_path}")
                print("Additional details:")
                print(f"Pixel mismatch: {current_image_path} (differs at coordinates {tuple(first_mismatch)})")
                all_identical = False

        return all_identical

    @staticmethod
    def verify_jsonl_identical(expected_jsonl: list[dict], current_jsonl: list[dict]) -> bool:
        """Check if json lines files are identical."""
        identical = True
        for expected_line, current_line in zip(expected_jsonl, current_jsonl):
            if expected_line != current_line:
                identical = False
                print(f"Expected: {expected_line} \n but found: {current_line}")
        return identical

    @staticmethod
    def clean_up(directory_to_erase: Path) -> None:
        """Recursively remove files from directory_to_erase."""
        files_to_erase = glob.glob(f"{str(directory_to_erase)}/**", recursive=True)
        dirs_to_erase = []

        # Remove files first
        for file in files_to_erase:
            if os.path.exists(file):
                if not os.path.isdir(file):
                    os.remove(file)
                else:
                    dirs_to_erase.append(os.path.join(file))
        # Sort longest to shortest to delete child directories first
        dirs_to_erase = sorted(dirs_to_erase, key=len, reverse=True)

        # Remove directories
        for directory in dirs_to_erase:
            os.rmdir(directory)
