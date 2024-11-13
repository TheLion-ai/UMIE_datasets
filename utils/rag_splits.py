"""Create train-val split while keeping same study_id together and stratifying based on source_labels."""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from jsonl file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def group_by_study_id(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group images by study_id."""
    study_groups = defaultdict(list)
    for item in data:
        study_groups[item["study_id"]].append(item)
    return study_groups


def get_study_label(study_group: List[Dict]) -> str:
    """Get the dominant source label for a study group."""
    label_counts: dict = defaultdict(int)
    for item in study_group:
        for label in item["source_labels"]:
            label_counts[label] += 1
    return max(label_counts.items(), key=lambda x: x[1])[0]


def calculate_val_size(study_df: pd.DataFrame, val_size: Union[float, int], total_images: int) -> float:
    """
    Calculate validation set size as a proportion based on input specification.

    Args:
        study_df: DataFrame containing study information
        val_size: Either a float (proportion) or int (absolute number)
        total_images: Total number of images in the dataset

    Returns:
        Float representing the proportion for validation set
    """
    if isinstance(val_size, float):
        if not 0 < val_size < 1:
            raise ValueError("Validation proportion must be between 0 and 1")
        return val_size
    elif isinstance(val_size, int):
        if not 0 < val_size < total_images:
            raise ValueError(f"Validation count must be between 0 and {total_images}")
        # Calculate the approximate proportion needed to get the desired number of examples
        return val_size / total_images
    else:
        raise ValueError("val_size must be either float (proportion) or int (count)")


def create_split(
    jsonl_path: str, val_size: Union[float, int] = 0.2, random_state: Optional[int] = 42
) -> Dict[str, List[Dict]]:
    """
    Create train-val split while keeping same study_id together and stratifying based on source_labels.

    Args:
        jsonl_path: Path to the jsonl file
        val_size: Either a float between 0-1 (proportion) or int (exact number of examples)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing lists of complete JSON records for train and val splits
    """
    # Load and group data
    data = load_jsonl(jsonl_path)
    study_groups = group_by_study_id(data)

    # Create a list of (study_id, dominant_label, group_size) tuples
    study_info = [(study_id, get_study_label(group), len(group)) for study_id, group in study_groups.items()]

    # Convert to DataFrame for easier manipulation
    study_df = pd.DataFrame(study_info, columns=["study_id", "label", "group_size"])

    total_images = sum(len(group) for group in study_groups.values())
    val_proportion = calculate_val_size(study_df, val_size, total_images)

    # Perform stratified split on study level
    train_studies, val_studies = train_test_split(
        study_df["study_id"], test_size=val_proportion, random_state=random_state, stratify=study_df["label"]
    )

    # Convert to sets for faster lookup
    train_studies_set = set(train_studies)
    # val_studies_set = set(val_studies)

    # Create lists of complete JSON records for each split
    train_records = []
    val_records = []

    for item in data:
        if item["study_id"] in train_studies_set:
            train_records.append(item)
        else:
            val_records.append(item)

    # Print split statistics
    print("\nSplit Statistics:")
    print(f"Total studies: {len(study_groups)}")
    print(f"Total images: {total_images}")
    print(f"Train studies: {len(train_studies)}")
    print(f"Val studies: {len(val_studies)}")
    print(f"Train images: {len(train_records)}")
    print(f"Val images: {len(val_records)}")

    # Validate if using exact number
    if isinstance(val_size, int):
        print(f"\nRequested validation size: {val_size}")
        print(f"Actual validation size: {len(val_records)}")
        print("Note: Actual size might differ slightly from requested size due to study grouping")

    return {"train": train_records, "val": val_records}


def save_split_files(splits: Dict[str, List[Dict]], output_dir: str) -> None:
    """Save train and val splits to separate JSONL files."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, records in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"\nSaved {split_name} split to {output_path}")
        print(f"Number of records in {split_name}: {len(records)}")


# Example usage
if __name__ == "__main__":
    jsonl_path = (
        "/home/basia/UMIE_datasets/cropped_data/03_brain_tumor_classification/03_brain_tumor_classification.jsonl"
    )
    output_dir = "/home/basia/UMIE_datasets/cropped_data/03_brain_tumor_classification"

    # Create splits
    splits = create_split(jsonl_path=jsonl_path, val_size=100, random_state=42)

    # Save splits to files
    save_split_files(splits, output_dir)
