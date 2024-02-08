"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""
from abc import abstractmethod

from src.constants import TARGET_PATH
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.brain_tumor_detection import BrainTumorDetectionPipeline
from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.kits21 import KITS21Pipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

datasets = [
    KITS21Pipeline(
        path_args={
            "source_path": "",
            "target_path": "./data/",
            "labels_path": "",
        },
    ),
    StanfordCOCAPipeline(
        path_args={
            "source_path": "",
            "target_path": "./data/",
            "masks_path": "",
        },
    ),
    StanfordBrainMETPipeline(
        path_args={
            "source_path": "",
            "target_path": "./data/",
        },
    ),
    CoronahackChestXrayPipeline(
        path_args={
            "source_path": "",
            "target_path": "./data/",
        },
    ),
    AlzheimersPipeline(
        path_args={
            "source_path": "/Users/piotr/Desktop/UMIE/data/Alzheimer_s Dataset",
            "target_path": "/Users/piotr/Desktop/UMIE/processed_data",
        },
    ),
    BrainTumorDetectionPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
]
