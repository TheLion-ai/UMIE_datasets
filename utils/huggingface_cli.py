"""Hugging Face CLI for uploading datasets to the Hub."""
import importlib.util
import json
import os

import click
from datasets import Image, Value, load_dataset

from config.dataset_config import all_datasets

hf_repo_name = "lion-ai/umie_datasets"


def to_absolute_path(relative_path: str, data_dir: str) -> str:
    """
    Convert a relative path to an absolute path by joining it with the data directory path.

    Args:
        relative_path (str): The relative path.
        data_dir (str): The data directory path.

    Returns:
        str: The absolute path.
    """
    return os.path.abspath(os.path.join(data_dir, relative_path))


def transform_labels(example: dict) -> dict:
    """
    Transform the labels in the example by converting them to a string format.

    Args:
        example (dict): The example containing the labels.

    Returns:
        dict: The example with transformed labels.
    """
    labels = example["labels"]
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError:
            # If it's not valid JSON, return it as is
            raise ValueError(f"Invalid JSON format: {labels}")

    new_labels = {}
    for label in labels:
        if isinstance(label, dict):
            for k, v in label.items():
                if v is not None:
                    new_labels[k] = v
        else:
            raise ValueError(f"Invalid label type: {type(label)}")
    return {"labels": json.dumps(new_labels)}


@click.command()
@click.option(
    "--dataset-name",
    prompt="Select dataset",
    help="Name of the dataset",
    type=click.Choice([dataset.dataset_name for dataset in all_datasets]),
)
@click.option(
    "--commit-message",
    prompt="Enter commit message",
    help="Commit message for the upload",
)
def upload(dataset_name: str, commit_message: str) -> None:
    """
    Upload a dataset to the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset.
        commit_message (str): The commit message for the upload.
    """
    dataset_config = [dataset for dataset in all_datasets if dataset.dataset_name == dataset_name][0]
    name_with_id = dataset_config.dataset_uid + "_" + dataset_config.dataset_name
    current_directory = os.getcwd()
    data_dir = os.path.join(current_directory, "data")
    datasets_dir = os.path.join(data_dir, name_with_id)
    dataset_file = os.path.join(datasets_dir, f"{name_with_id}.jsonl")

    if not os.path.exists(datasets_dir):
        raise FileNotFoundError(f"Data directory {datasets_dir} does not exist")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset jsonl file {dataset_file} does not exist")

    dataset = load_dataset("json", data_files={dataset_name: dataset_file})[dataset_name]

    # Rename columns
    dataset = dataset.rename_column("umie_path", "image")
    dataset = dataset.rename_column("mask_path", "mask")

    # convert image mask columns from string to Image type
    dataset = dataset.map(lambda example: {"image": to_absolute_path(example["image"], data_dir)})
    dataset = dataset.cast_column("image", Image())

    if dataset_config.masks != {}:
        dataset = dataset.map(lambda example: {"mask": to_absolute_path(example["mask"], data_dir)})
        dataset = dataset.cast_column("mask", Image())
    else:
        dataset = dataset.map(lambda example: {"mask": None})

    # convert labels to string dump of dictionary
    label_schema = Value("string")
    dataset = dataset.map(transform_labels)
    new_features = dataset.features.copy()
    new_features["labels"] = label_schema
    dataset = dataset.cast(new_features)

    dataset.push_to_hub(
        hf_repo_name,
        dataset_name,
        create_pr=True,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    # Add the main CLI group and commands
    @click.group()
    def cli() -> None:
        """Hugging Face CLI for uploading datasets to the Hub."""
        pass

    cli.add_command(upload)

    cli()
