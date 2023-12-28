"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""
from abc import abstractmethod

from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.kits21 import KITS21Pipeline
from src.pipelines.lidcidri import LidcIdriPipeline
from src.pipelines.lidcidri import LidcIdriPipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

datasets = [
    KITS21Pipeline(
        path_args={
            "source_path": "",
            "source_path": "",
            "target_path": "./data/",
            "labels_path": "",
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
            "source_path": "/home/basia/Desktop/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0002",
            "target_path": "./data/",
        },
    ),
    LidcIdriPipeline(
        path_args={
            "source_path": "/home/basia/Desktop/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0002",
            "target_path": "./data/",
            "masks_path": "/home/basia/Desktop/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0002",
        },
    ),
]
