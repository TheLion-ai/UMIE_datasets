"""
test_brain_with_hemorrhage.

Objective:
This test checks whether Pipeline for Brain With Hemorrhage dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.brain_with_hemorrhage import BrainWithHemorrhagePipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(
    os.getcwd(),
    "testing/test_dummy_data/19_BrainHemorrhage/input/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
)
masks_path = os.path.join(
    os.getcwd(),
    "testing/test_dummy_data/19_BrainHemorrhage/input/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
)
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/19_BrainHemorrhage/output")
expected_output_path = os.path.join(os.getcwd(), "testing/test_dummy_data/19_BrainHemorrhage/expected_output")


def test_run_brain_with_hemorrhage():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = BrainWithHemorrhagePipeline(
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
        pytest.fail(f'Trying to run Brain with Hemorrhage pipeline raised an exception: "{e}"')


def test_brain_with_hemorrhage_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("Brain wih hemorrhage pipeline created file tree different than expected.")


def test_brain_with_hemorrhage_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("Brain wih hemorrhage pipeline created image contents different than expected.")


def test_clean_up_brain_with_hemorrhage():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
