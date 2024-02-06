"""
test_stanford_coca.

Objective:
This test checks whether the Pipeline for Stanford COCA dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.stanford_coca import StanfordCOCAPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(
    os.getcwd(),
    "testing/test_dummy_data/15_StanfordCOCA/input/cocacoronarycalciumandchestcts-2/Gated_release_final/patient",
)
masks_path = os.path.join(
    os.getcwd(),
    "testing/test_dummy_data/15_StanfordCOCA/input/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml",
)
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/15_StanfordCOCA/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/15_StanfordCOCA/expected_output")


def test_run_stanford_coca():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = StanfordCOCAPipeline(
        path_args={"source_path": source_path, "target_path": target_path, "masks_path": masks_path},
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Stanford COCA pipeline raised an exception: "{e}"')


def test_stanford_coca_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Stanford COCA pipeline created file tree different than expected.")


def test_stanford_coca_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Stanford COCA pipeline created image contents different than expected.")


def test_clean_up_coronahack():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
