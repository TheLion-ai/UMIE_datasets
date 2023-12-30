"""
test_check_runner_config_paths.

Objective:
This test checks whether runner_config.py doesn't include unwanted local paths,
and includes default target path.
"""
from config.runner_config import datasets
from src.constants import TARGET_PATH


def test_target_paths_default():
    """Test that checks if target_path is set to default value."""
    for dataset in datasets:
        assert dataset.path_args["target_path"] == TARGET_PATH


def test_nontarget_paths_empty():
    """Test that checks if paths other than target_path are empty."""
    for dataset in datasets:
        for path_type in dataset.path_args.keys():
            if path_type == "target_path":
                continue
            assert dataset.path_args[path_type] == ""
