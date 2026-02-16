"""
test_check_pneumothorax.

Objective:
This test checks whether Pipeline for Pneumothorax dataset runs correctly.
"""

import glob
import json
import os

import pytest

from src.base.pipeline import PathArgs
from src.pipelines.pneumothorax import PneumothoraxPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/29_pneumothorax/input/siim-acr-pneumothorax")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/29_pneumothorax/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/29_pneumothorax/expected_output")


def test_initial_clean_up_pneumothorax():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)


def test_run_pneumothorax():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = PneumothoraxPipeline(
        path_args=PathArgs(
            source_path=source_path,
            labels_path=source_path,
            masks_path=source_path,
            target_path=target_path,
        ),
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Pneumothorax pipeline raised an exception: "{e}"')


def test_pneumothorax_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Pneumothorax's pipeline created file tree different than expected.")


def test_pneumothorax_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Pneumothorax's pipeline created image contents different than expected.")


def test_pneumothorax_verify_jsonl_correct():
    """Test to verify whether all json files have contents as expected."""
    expected_jsonl_files = glob.glob(f"{expected_output_path}/**/**.jsonl", recursive=True)
    current_jsonl_files = glob.glob(f"{target_path}/**/**.jsonl", recursive=True)

    if not expected_jsonl_files:
        pytest.fail("No expected jsonl files found")
    if not current_jsonl_files:
        pytest.fail("No current jsonl files found")

    expected_jsonl_path = expected_jsonl_files[0]
    current_jsonl_path = current_jsonl_files[0]
    with open(expected_jsonl_path, "r") as file:
        expected_jsonl = [json.loads(line) for line in file]
    with open(current_jsonl_path, "r") as file:
        current_jsonl = [json.loads(line) for line in file]

    if not DatasetTestingLibrary.verify_jsonl_identical(expected_jsonl, current_jsonl):
        pytest.fail("Pneumothorax's pipeline created json contents different than expected.")


def test_clean_up_pneumothorax():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
