"""
test_check_runner_config_paths.

Objective:
This test checks whether runner_config.py doesn't include unwanted local paths,
and includes default target path.
"""

import pytest

from config.runner_config import datasets
from src.constants import TARGET_PATH


def test_runner_config_target_paths_default():
    """Test that checks if target_path is set to default value."""
    for dataset in datasets:
        if dataset.path_args["target_path"] != TARGET_PATH:
            pytest.fail("Any of target_paths is not set to default value.")


def test_runner_config_nontarget_paths_empty():
    """Test that checks if paths other than target_path are empty."""
    for dataset in datasets:
        for path_type in dataset.path_args.keys():
            if path_type == "target_path":
                continue
            if dataset.path_args[path_type] != "":
                pytest.fail("Any of paths other than target_path is not set to empty value.")
