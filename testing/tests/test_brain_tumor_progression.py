"""
test_brain_tumor_progression.

Objective:
This test checks whether Pipeline for Brain Tumor Progression dataset runs correctly.
"""

import glob
import json
import os

import pytest
from zipp import Path

from base.pipeline import PathArgs
from src.pipelines.brain_tumor_progression import BrainTumorProgressionPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(
    os.getcwd(), "testing/test_dummy_data/10_brain_tumor_progression/input/brain_tumor_progression"
)
masks_path = os.path.join(
    os.getcwd(), "testing/test_dummy_data/10_brain_tumor_progression/input/brain_tumor_progression"
)
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/10_brain_tumor_progression/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/10_brain_tumor_progression/expected_output")

def test_run_brain_tumor_progression():
    """Test to verify, that there are no exceptions while running pipeline."""
    DatasetTestingLibrary.clean_up(target_path)
    dataset = BrainTumorProgressionPipeline(
        path_args=PathArgs(
            source_path=source_path,
            masks_path=masks_path,
            target_path=target_path,
        ),
    )
    pipeline = dataset.pipeline
    try:
        pipeline.transform(dataset.args["source_path"])
    except Exception as e:
        pytest.fail(f'Trying to run Brain Tumor Progression pipeline raised an exception: "{e}"')


def test_brain_tumor_progression_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Brain Tumor Progression pipeline created file tree different than expected.")


def test_brain_tumor_progression_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Brain Tumor Progression pipeline created image contents different than expected.")


def test_brain_tumor_progression_verify_jsonl_correct():
    """Test to verify whether all json files have contents as expected."""
    expected_jsonl_path = glob.glob(f"{expected_output_path}/**/**.jsonl", recursive=True)[0]
    current_jsonl_path = glob.glob(f"{target_path}/**/**.jsonl", recursive=True)[0]
    with open(expected_jsonl_path, "r") as file:
        expected_jsonl = [json.loads(line) for line in file]
    with open(current_jsonl_path, "r") as file:
        current_jsonl = [json.loads(line) for line in file]

    if not DatasetTestingLibrary.verify_jsonl_identical(expected_jsonl, current_jsonl):
        pytest.fail("Brain Tumor Progression pipeline created jsonl contents different than expected.")


def test_clean_up_brain_tumor_progression():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
