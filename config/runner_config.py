"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""
from abc import abstractmethod

from src.pipelines.brain_with_hemorrhage import BrainWithHemorrhagePipeline
from src.pipelines.kits21 import KITS21Pipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

datasets = [
    BrainWithHemorrhagePipeline(
        path_args={
            "source_path": "",
            "target_path": "./data/",
            "masks_path": "",
        },
    ),
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
]
