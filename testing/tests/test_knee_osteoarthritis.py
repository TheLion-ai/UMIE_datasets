"""
test_check_knee_osteoarthritis.

Objective:
This test checks whether Pipeline for Knee Osteoarthritis dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.knee_osteoarthritis import KneeOsteoarthritisPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/10_knee_osteoarthritis/input/Knee_Osteoarthritis")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/10_knee_osteoarthritis/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/10_knee_osteoarthritis/expected_output")


def test_run_knee_osteoarthritis():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = KneeOsteoarthritisPipeline(
        path_args={
            "source_path": source_path,
            "target_path": target_path,
        },
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Knee Osteoarthritis pipeline raised an exception: "{e}"')


def test_knee_osteoarthritis_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Knee Osteoarthritis pipeline created file tree different than expected.")


def test_knee_osteoarthritis_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Knee Osteoarthritis pipeline created image contents different than expected.")


def test_clean_up_knee_osteoarthritis():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
