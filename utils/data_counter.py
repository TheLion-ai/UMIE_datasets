"""Count the number of labels and masks in the dataset and calculate the weights for the loss function."""

import json
import os
from calendar import c
from logging import config

import cv2
import jsonlines
import numpy as np

from config.dataset_config import all_datasets
from config.labels import all_labels
from config.masks import all_masks
from src.constants import TARGET_PATH


def delete_blank_images() -> None:
    """Delete blank images from the dataset."""
    for dataset in all_datasets:
        if dataset.dataset_name not in ["cmmd"]:
            continue
        print(f"Checking {dataset.dataset_name}...")
        name_with_id = dataset.dataset_uid + "_" + dataset.dataset_name
        current_directory = os.getcwd()
        data_dir = os.path.join(current_directory, "data")
        datasets_dir = os.path.join(data_dir, name_with_id)
        dataset_file = os.path.join(datasets_dir, f"{name_with_id}.jsonl")

        non_blank_images = []
        with jsonlines.open(dataset_file, mode="r") as reader:
            for obj in reader:
                img_path = os.path.join(TARGET_PATH, obj["umie_path"])
                img = cv2.imread(img_path)
                if np.unique(img).shape[0] == 1:
                    os.remove(img_path)
                    print(f"Deleted {img_path}")
                else:
                    non_blank_images.append(obj)
        with jsonlines.open(dataset_file, mode="w") as writer:
            for obj in non_blank_images:
                writer.write(obj)


def calculate_labels_and_mask_ratios() -> None:
    """Calculate the number of labels and masks in the dataset."""
    label_counts = {}
    mask_counts = {}
    all_imgs_count = 0
    for label in all_labels:
        label_counts[label.radlex_name] = 0
    for mask in all_masks:
        mask_counts[mask.radlex_name] = 0
    for dataset in all_datasets:
        print(f"Checking {dataset.dataset_name}...")
        name_with_id = dataset.dataset_uid + "_" + dataset.dataset_name
        current_directory = os.getcwd()
        data_dir = os.path.join(current_directory, "data")
        datasets_dir = os.path.join(data_dir, name_with_id)
        dataset_file = os.path.join(datasets_dir, f"{name_with_id}.jsonl")

        with jsonlines.open(dataset_file, mode="r") as reader:
            for obj in reader:
                if len(obj["labels"]) >= 0:
                    for label in obj["labels"]:
                        label = list(label.keys())[0]
                        label_counts[label] += 1
                if obj["mask_path"]:
                    mask_path = os.path.join(TARGET_PATH, obj["mask_path"])
                    mask = cv2.imread(mask_path)
                    for color in list(np.unique(mask)):
                        if color != 0:
                            for mask in all_masks:
                                if mask.color == color:
                                    mask_counts[mask.radlex_name] += 1
                all_imgs_count += 1

    print(f"Total images: {all_imgs_count}")
    print("Labels:")
    for label in label_counts:
        print(f"{label}: {label_counts[label]}")
    print("Masks:")
    for mask in mask_counts:
        print(f"{mask}: {mask_counts[mask]}")
    counts = {"label_counts": label_counts, "mask_counts": mask_counts, "all_imgs_count": all_imgs_count}
    with open("data/counts.json", "w") as f:
        json.dump(counts, f)


def calculate_weights_pytorch() -> None:
    """Calculate the weights for the loss function."""
    pos_weights: dict = {"label_counts": {}, "mask_counts": {}}
    with open("data/counts.json", "r") as f:
        counts = json.load(f)
    for label in counts["label_counts"].keys():
        neg = counts["label_counts"][label]
        if neg == 0:
            pos_weights["label_counts"][label] = 0
            continue
        pos = counts["all_imgs_count"] - neg
        pos_weights["label_counts"][label] = neg / pos
    for mask in counts["mask_counts"].keys():
        neg = counts["mask_counts"][mask]
        if neg == 0:
            pos_weights["mask_counts"][mask] = 0
            continue
        pos = counts["all_imgs_count"] - neg
        pos_weights["mask_counts"][mask] = neg / pos

    label_counts_list = []
    for label in all_labels:
        label_counts_list.append(pos_weights["label_counts"][label.radlex_name])
    pos_weights["label_counts_list"] = label_counts_list
    mask_counts_list = []
    for mask in all_masks:
        mask_counts_list.append(pos_weights["mask_counts"][mask.radlex_name])
    pos_weights["mask_counts_list"] = mask_counts_list

    with open("data/pos_weights.json", "w") as f:
        json.dump(pos_weights, f)


if __name__ == "__main__":
    # delete_blank_images()
    calculate_labels_and_mask_ratios()
    calculate_weights_pytorch()
