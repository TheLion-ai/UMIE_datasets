"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""
from abc import abstractmethod

from src.constants import TARGET_PATH
from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.kits23 import KITS23Pipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

datasets = [
    KITS23Pipeline(
        path_args={
            "source_path": "/home/basia/Desktop/UMIE_source/kits23_test",  # Path to the dataset directory in KITS23 repo
            "target_path": TARGET_PATH,
            "labels_path": "/home/basia/Desktop/UMIE_source/kits23_test/kits23.json",  # Path to kits.json
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
]
