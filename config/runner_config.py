"""Config file with the paths to the datasets and their parameters. The user defines the paths and parameters here for the datasets of his interest."""

from src.constants import TARGET_PATH
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.brain_tumor_classification import BrainTumorClassificationPipeline
from src.pipelines.alzheimers import AlzheimersPipeline
from src.pipelines.brain_met_share import BrainMETSharePipeline
from src.pipelines.brain_tumor_detection import BrainTumorDetectionPipeline
from src.pipelines.brain_with_intracranial_hemorrhage import (
    BrainWithIntracranialHemorrhagePipeline,
)
from src.pipelines.chest_xray14 import ChestXray14Pipeline
from src.pipelines.coca import COCAPipeline
from src.pipelines.coronahack import CoronaHackPipeline
from src.pipelines.covid19_detection import COVID19DetectionPipeline
from src.pipelines.finding_and_measuring_lungs import FindingAndMeasuringLungsPipeline
from src.pipelines.kits23 import KITS23Pipeline
from src.pipelines.knee_osteoarthritis import KneeOsteoarthritisPipeline
from src.pipelines.lits import LITSPipeline

datasets = [
    KITS23Pipeline(
        path_args={
            "source_path": "",  # Path to the "dataset" directory in KITS23 repo
            "masks_path": "",  # Path to the "dataset" directory in KITS23 repo
            "target_path": TARGET_PATH,
            "labels_path": "",  # Path to kits23.json
        },
    ),
    CoronaHackPipeline(
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
    COVID19DetectionPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    FindingAndMeasuringLungsPipeline(
        path_args={
            "source_path": "",  # Path to 2d_images directory
            "target_path": TARGET_PATH,
            "masks_path": "",  # Path to 2d_masks directory
        },
    ),
    BrainTumorDetectionPipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    BrainWithIntracranialHemorrhagePipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
            "masks_path": "",
        },
    ),
    ChestXray14Pipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
       
        },
    ),
    BrainMETSharePipeline(
        path_args={
            "source_path": "",
            "target_path": TARGET_PATH,
        },
    ),
    COCAPipeline(
        path_args={
            "source_path": "",  # Path to Gated_release_final/patient
            "target_path": TARGET_PATH,
            "masks_path": "",  # Path to Gated_release_final/calcium_xml
        },
    ),
    LITSPipeline(
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
