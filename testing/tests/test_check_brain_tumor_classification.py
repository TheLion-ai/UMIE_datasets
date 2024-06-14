"""
test_check_brain_tumor_classification.

Objective:
This test checks whether Pipeline for Brain Tumor Classification dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.brain_tumor_classification import BrainTumorClassificationPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/03_brain_tumor_classification/input")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/03_brain_tumor_classification/output")
expected_output_path = os.path.join(
    os.getcwd(), "testing/test_dummy_data/03_brain_tumor_classification/expected_output"
)


def test_run_brain_tumor_classification():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = BrainTumorClassificationPipeline(
        path_args={
            "source_path": source_path,
            "target_path": target_path,
        },
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run brain_tumor_classification pipeline raised an exception: "{e}"')


def test_brain_tumor_classification_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Brain Tumor Classification pipeline created file tree different than expected.")


def test_brain_tumor_classification_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Brain Tumor Classification pipeline created image contents different than expected.")


def test_clean_up_brain_tumor_classification():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
