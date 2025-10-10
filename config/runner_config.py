"""
Config file with the paths to the datasets and their parameters.

The user fills in the paths for the datasets they want to transform.
The dataset with empty source_path is not transformed.
If the dataset has a labels_path or masks path, it needs to be filled in too in order to transform the dataset.
Each dataset should have a comment suggesting the name of the source file the path should point to, e.g. # Path to kits23.json.
"""
import os

from dotenv import load_dotenv

from src.base.pipeline import PathArgs
from src.constants import TARGET_PATH
from src.pipelines import (
    AlzheimersPipeline,
    BrainMETSharePipeline,
    BrainTumorClassificationPipeline,
    BrainTumorDetectionPipeline,
    BrainTumorProgressionPipeline,
    BrainWithIntracranialHemorrhagePipeline,
    ChestXray14Pipeline,
    CmmdPipeline,
    COCAPipeline,
    CoronaHackPipeline,
    COVID19DetectionPipeline,
    CtOrgPipeline,
    FindingAndMeasuringLungsPipeline,
    KITS23Pipeline,
    KneeOsteoarthritisPipeline,
    LidcIdriPipeline,
    LITSPipeline,
)

load_dotenv(dotenv_path=".pipeline.env")

datasets = [
    KITS23Pipeline(
        path_args=PathArgs(
            source_path=os.getenv("KITS23", ""),  # Path to the "dataset" directory in KITS23 repo
            masks_path=os.getenv("KITS23", ""),  # Path to the "dataset" directory in KITS23 repo
            target_path=TARGET_PATH,
            labels_path=os.getenv("KITS23", "") + "/kits23.json",  # Path to kits23.json
        ),
    ),
    CoronaHackPipeline(
        path_args=PathArgs(
            source_path=os.getenv("CORONA_HACK", ""),
            target_path=TARGET_PATH,
            labels_path=os.getenv("CORONA_HACK", "")
            + "/Chest_xray_Corona_Metadata.csv",  # Path to Chest_xray_Corona_Metadata.csv
        ),
    ),
    AlzheimersPipeline(
        path_args=PathArgs(
            source_path=os.getenv("ALZHEIMER", ""),  # Path to archive directory from kaggle
            target_path=TARGET_PATH,
        ),
    ),
    BrainTumorClassificationPipeline(
        path_args=PathArgs(
            source_path=os.getenv("BRAIN_TUMOR", ""),
            target_path=TARGET_PATH,
        ),
    ),
    COVID19DetectionPipeline(
        path_args=PathArgs(
            source_path=os.getenv("COVID19_DETECTION", ""),
            target_path=TARGET_PATH,
        ),
    ),
    FindingAndMeasuringLungsPipeline(
        path_args=PathArgs(
            source_path=os.getenv("FAM_LUNGS", ""),  # Path to 2d_images directory
            target_path=TARGET_PATH,
            masks_path=os.getenv("FAM_LUNGS", ""),  # Path to 2d_masks directory
        ),
    ),
    BrainWithIntracranialHemorrhagePipeline(
        path_args=PathArgs(
            source_path=os.getenv("BRAIN_CT", ""),
            target_path=TARGET_PATH,
            masks_path=os.getenv("BRAIN_CT", ""),  # same as source path
        ),
    ),
    LITSPipeline(
        path_args=PathArgs(
            source_path=os.getenv("LITS", ""),
            target_path=TARGET_PATH,
            masks_path=os.getenv("LITS", ""),  # same as source_path
        ),
    ),
    BrainTumorDetectionPipeline(
        path_args=PathArgs(
            source_path=os.getenv("BRAIN_MRI", ""),
            target_path=TARGET_PATH,
        ),
    ),
    KneeOsteoarthritisPipeline(
        path_args=PathArgs(
            source_path=os.getenv("KNEE_OSTEO", ""),
            target_path=TARGET_PATH,
        ),
    ),
    # BrainTumorProgressionPipeline( # data collection removed
    #     path_args=PathArgs(
    #         source_path="",
    #         target_path=TARGET_PATH,
    #         masks_path="",
    #     )
    # ),
    ChestXray14Pipeline(
        path_args=PathArgs(
            source_path=os.getenv("CHESTXRAY14", "") + "/images",  # path to images/
            target_path=TARGET_PATH,
            labels_path=os.getenv("CHESTXRAY14", "")
            + "/Data_Entry_2017_v2020.csv",  # Path to Data_Entry_2017_v2020.csv
        ),
    ),
    # unverified datasets
    # COCAPipeline(
    #     path_args=PathArgs(
    #         source_path="",  # Path to Gated_release_final/patient
    #         target_path=TARGET_PATH,
    #         masks_path="",  # Path to Gated_release_final/calcium_xml
    #     ),
    # ),
    # BrainMETSharePipeline(
    #     path_args=PathArgs(
    #         source_path="",
    #         target_path=TARGET_PATH,
    #     ),
    # ),
    # CtOrgPipeline(
    #     path_args=PathArgs(
    #         source_path="",
    #         target_path=TARGET_PATH,
    #         masks_path="",
    #     ),
    # ),
    LidcIdriPipeline(
        path_args=PathArgs(
            source_path=os.getenv("LIDC_MANIFEST", ""),  # Path to LIDC-IDRI/ directory
            target_path=TARGET_PATH,
            masks_path=os.getenv("LIDC_XML", ""),  # Path to extracted LIDC-XML-only/ directory (from LIDC-XML-only.zip)
        ),
    ),
    CmmdPipeline(
        path_args=PathArgs(
            source_path=os.getenv("CMMD_MANIFEST", ""),  # Path to 'manifest-{xxxxxxxxxxxxx}/CMMD' folder
            target_path=TARGET_PATH,
            labels_path=os.getenv("CMMD_CLINICAL", ""),  # Path to 'CMMD_clinicaldata_revision.xlsx' file
        ),
    ),
]
