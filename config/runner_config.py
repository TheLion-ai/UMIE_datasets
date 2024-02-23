"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""
from abc import abstractmethod

from src.constants import TARGET_PATH
from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.covid19_detection import Covid19Detection
from src.pipelines.kits23 import KITS23Pipeline
from src.pipelines.knee_osteoarthritis import KneeOsteoarthritisPipeline
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
            "masks_path": "",  # Path to Gated_release_final/calcium_xml
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
    KneeOsteoarthritisPipeline(
        path_args={
            "source_path": "/Users/piotr/Desktop/UMIE/data/Knee_Osteoarthritis",
            "target_path": "/Users/piotr/Desktop/UMIE/processed_data",
        },
    ),
]
