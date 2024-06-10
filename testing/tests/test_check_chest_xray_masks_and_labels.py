"""
test_chest_xray_masks_and_labels.

Objective:
This test checks whether Pipeline for test_chest_xray_masks_and_labels dataset runs correctly.
"""

import glob
import os

import pytest

from src.pipelines.chest_xray_masks_and_labels import LungSegmentationPipeline
from testing.libs.dataset_testing_lib import DatasetTestingLibrary

source_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_Lung_segmentation_from_Chest_X-Rays/input")
target_path = os.path.join(os.getcwd(), "testing/test_dummy_data/08_Lung_segmentation_from_Chest_X-Rays/output")
masks_path = os.path.join(
    os.getcwd(), "testing/test_dummy_data/08_Lung_segmentation_from_Chest_X-Rays/input/ManualMask/"
)
expected_output_path = os.path.join(
    os.getcwd(), "testing/test_dummy_data/08_Lung_segmentation_from_Chest_X-Rays/expected_output"
)


def test_run_chest_xray_masks_and_labels():
    """Test to verify, that there are no exceptions while running pipeline."""
    dataset = LungSegmentationPipeline(
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
        pytest.fail(f'Trying to run Lung Segmentation pipeline raised an exception: "{e}"')


def test_chest_xray_masks_and_labels_verify_file_tree():
    """Test to verify if file tree is as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**", recursive=True)

    if not DatasetTestingLibrary.verify_file_tree(expected_file_tree, current_file_tree):
        pytest.fail("chest_xray_masks_and_labels pipeline created file tree different than expected.")


def test_chest_xray_masks_and_labels_verify_images_correct():
    """Test to verify whether all images have contents as expected."""
    expected_file_tree = glob.glob(f"{str(expected_output_path)}/**/*.png", recursive=True)
    current_file_tree = glob.glob(f"{str(target_path)}/**/*.png", recursive=True)

    if not DatasetTestingLibrary.verify_all_images_identical(expected_file_tree, current_file_tree):
        pytest.fail("chest_xray_masks_and_labels pipeline created image contents different than expected.")


def test_clean_up_chest_xray_masks_and_labels():
    """Removes output folder with it's contents."""
    DatasetTestingLibrary.clean_up(target_path)
