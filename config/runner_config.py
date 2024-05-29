"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""

from src.constants import TARGET_PATH
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.brain_tumor_classification import BrainTumorClassificationPipeline
from src.pipelines.brain_tumor_detection import BrainTumorDetectionPipeline
from src.pipelines.brain_with_hemorrhage import BrainWithHemorrhagePipeline
from src.pipelines.chest_xray14 import ChestXray14Pipeline
from src.pipelines.coronahack_chest_xray import CoronahackChestXrayPipeline
from src.pipelines.covid19_detection import Covid19Detection
from src.pipelines.finding_and_measuring_lungs_in_ct import (
    FindingAndMeasuringLungsInCTPipeline,
)
from src.pipelines.kits23 import KITS23Pipeline
from src.pipelines.knee_osteoarthritis import KneeOsteoarthritisPipeline
from src.pipelines.liver_and_liver_tumor_pipeline import LiverAndLiverTumorPipeline
from src.pipelines.stanford_brain_met import StanfordBrainMETPipeline
from src.pipelines.stanford_coca import StanfordCOCAPipeline

from config import dataset_config

datasets = [
    KITS23Pipeline(
        path_args={
            "source_path": "/home/basia/kits_sample/case_00000",  # Path to the dataset directory in KITS23 repo
            "target_path": TARGET_PATH,
            "masks_path": "/home/basia/kits_sample/case_00000",
            "labels_path": "/home/basia/kits_sample/kits23.json",  # Path to kits23.json
        },
        dataset_args=dataset_config.KITS23
    ),
    StanfordCOCAPipeline(
        path_args={
            "source_path": "",  # Path to Gated_release_final/patient
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
        dataset_args=dataset_config.StanfordCOCA
    ),
    FindingAndMeasuringLungsInCTPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
        dataset_args=dataset_config.Finding_and_Measuring_Lungs_in_CT_Data
    ),
    StanfordBrainMETPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
        dataset_args=dataset_config.StanfordBrainMET
    ),
    CoronahackChestXrayPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
        dataset_args=dataset_config.CoronaHack_Chest_XRay
    ),
    # AlzheimersPipeline(
    #     path_args={
    #         "source_path": "",
    #         "target_path": TARGET_PATH,
    #     },
    #     dataset_args=dataset_config.Alzheimers_Dataset
    # ),
    BrainTumorDetectionPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
        dataset_args=dataset_config.Brain_Tumor_Detection
    ),
    Covid19Detection(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
        dataset_args=dataset_config.Covid19_Detection
    ),
    ChestXray14Pipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
        dataset_args=dataset_config.ChestXray14
    ),
    BrainWithHemorrhagePipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
        dataset_args=dataset_config.Brain_with_hemorrhage
    ),
    LiverAndLiverTumorPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
    ),
    KneeOsteoarthritisPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        }
    ),
    BrainTumorClassificationPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
]
