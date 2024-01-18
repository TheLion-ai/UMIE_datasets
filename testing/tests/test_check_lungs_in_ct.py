"""
test_coronahack.

Objective:
This test checks whether Pipeline for Lungs In CT dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.finding_and_measuring_lungs_in_ct import (
    FindingAndMeasuringLungsInCTPipeline,
)
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/17_LungsInCT/input/archive/2d_images")
masks_path = os.path.join(os.getcwd(), "testing/test_dummy_data/17_LungsInCT/input/archive/2d_masks")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/17_LungsInCT/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/17_LungsInCT/expected_output")


def test_run_coronahack():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = FindingAndMeasuringLungsInCTPipeline(
        path_args={
            "source_path": source_path,
            "masks_path": masks_path,
            "target_path": target_path,
        },
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Lungs In CT pipeline raised an exception: "{e}"')


def test_lungs_in_ct_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Lungs In CT pipeline created file tree different than expected.")


def test_lungs_in_ct_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Lungs In CT pipeline created image contents different than expected.")


def test_clean_up_lungs_in_ct():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
