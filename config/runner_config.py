"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""

from abc import abstractmethod

from src.constants import TARGET_PATH
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.brain_tumor_detection import BrainTumorDetectionPipeline
from src.pipelines.brain_with_hemorrhage import BrainWithHemorrhagePipeline
from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.covid19_detection import Covid19Detection
from src.pipelines.finding_and_measuring_lungs_in_ct import (
    FindingAndMeasuringLungsInCTPipeline,
)
from src.pipelines.kits23 import KITS23Pipeline
from src.pipelines.liver_and_liver_tumor_pipeline import LiverAndLiverTumorPipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

datasets = [
    KITS23Pipeline(
        path_args={
            "source_path": "",  # Path to the dataset directory in KITS23 repo
            "target_path": TARGET_PATH,
            "labels_path": "",  # Path to kits23.json
        },
    ),
    StanfordCOCAPipeline(
        path_args={
            "source_path": "",  # Path to Gated_release_final/patient
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
    ),
    FindingAndMeasuringLungsInCTPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
    ),
    StanfordBrainMETPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    CoronahackChestXrayPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    AlzheimersPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    BrainTumorDetectionPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    Covid19Detection(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    BrainWithHemorrhagePipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
    ),
    LiverAndLiverTumorPipeline(
        path_args={
            "source_path": "/Users/piotr/Desktop/UMIE/data/liver_and_liver_tumor/dataset_6/dataset_6",
            "target_path": "/Users/piotr/Desktop/UMIE/processed_data",
            "masks_path": "/Users/piotr/Desktop/UMIE/data/liver_and_liver_tumor/dataset_6/dataset_6",
        },
    ),
]
