"""
test_liver_and_liver_tumor.

Objective:
This test checks whether the Pipeline for Liver_And_Liver_Tumor detection dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.lits import LITSPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_lits/input")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_lits/output")
<<<<<<< HEAD
masks_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_lits/input")
=======
>>>>>>> 5db70c6 (Add new coding for lits and knee)
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_lits/expected_output")


def test_run_lits():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = LITSPipeline(
        path_args={
            "source_path": source_path,
            "target_path": target_path,
            "masks_path": masks_path,
        },
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Liver_and_liver_tumor pipeline raised an exception: "{e}"')


def test_lits_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Liver_and_liver_tumor pipeline created file tree different than expected.")


def test_lits_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Liver_and_liver_tumor pipeline created image contents different than expected.")


def test_clean_up_lits():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
