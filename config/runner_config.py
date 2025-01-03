"""
Config file with the paths to the datasets and their parameters.

The user fills in the paths for the datasets they want to transform.
The dataset with empty source_path is not transformed.
If the dataset has a labels_path or masks path, it needs to be filled in too in order to transform the dataset.
Each dataset should have a comment suggesting the name of the source file the path should point to, e.g. # Path to kits23.json.
"""
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

datasets = [
    KITS23Pipeline(
        path_args=PathArgs(
            source_path="",  # Path to the "dataset" directory in KITS23 repo
            masks_path="",  # Path to the "dataset" directory in KITS23 repo
            target_path=TARGET_PATH,
            labels_path="",  # Path to kits23.json
        ),
    ),
    CoronaHackPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
            labels_path="",  # Path to Chest_xray_Corona_Metadata.csv
        ),
    ),
    # AlzheimersPipeline(
    #     path_args=PathArgs(
    #         source_path="/mnt/data/UMIE_source_data/02_alzheimers",  # Path to archive directory from kaggle
    #         target_path=TARGET_PATH,
    #     ),
    # ),
    BrainTumorClassificationPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
        ),
    ),
    COVID19DetectionPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
        ),
    ),
    FindingAndMeasuringLungsPipeline(
        path_args=PathArgs(
            source_path="",  # Path to 2d_images directory
            target_path=TARGET_PATH,
            masks_path="",  # Path to 2d_masks directory
        ),
    ),
    BrainWithIntracranialHemorrhagePipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
            masks_path="",  # same as source path
        ),
    ),
    LITSPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
            masks_path="",  # same as source_path
        ),
    ),
    BrainTumorDetectionPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
        ),
    ),
    KneeOsteoarthritisPipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/09_knee_osteoarthrithis",
            target_path=TARGET_PATH,
        ),
    ),
    BrainTumorProgressionPipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/10_brain_tumor_progression/brain_tumor_progression",
            target_path=TARGET_PATH,
            masks_path="/mnt/data/UMIE_source_data/10_brain_tumor_progression/brain_tumor_progression",
        )
    ),
    ChestXray14Pipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/11_chest_xray14/images",  # path to images/
            target_path=TARGET_PATH,
            labels_path="/mnt/data/UMIE_source_data/11_chest_xray14/Data_Entry_2017_v2020.csv",  # Path to Data_Entry_2017_v2020.csv
        ),
    ),
    COCAPipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/12_coca/cocacoronarycalciumandchestcts-2/Gated_release_final/patient",  # Path to Gated_release_final/patient
            target_path=TARGET_PATH,
            masks_path="/mnt/data/UMIE_source_data/12_coca/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml",  # Path to Gated_release_final/calcium_xml
        ),
    ),
    BrainMETSharePipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
        ),
    ),
    CtOrgPipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/14_ct_org/CT-ORG",
            target_path=TARGET_PATH,
            masks_path="/mnt/data/UMIE_source_data/14_ct_org/CT-ORG",
        ),
    ),
    LidcIdriPipeline(
        path_args=PathArgs(
            source_path="",  # Path to LIDC-IDRI/ directory
            target_path=TARGET_PATH,
            masks_path="",  # Path to extracted LIDC-XML-only/ directory (from LIDC-XML-only.zip)
        ),
    ),
    CmmdPipeline(
        path_args=PathArgs(
            source_path="/mnt/data/UMIE_source_data/18_cmmd/manifest-1616439774456/CMMD",  # Path to 'manifest-{xxxxxxxxxxxxx}/CMMD' folder
            target_path=TARGET_PATH,
            labels_path="/mnt/data/UMIE_source_data/18_cmmd/CMMD_clinicaldata_revision.xlsx",  # Path to 'CMMD_clinicaldata_revision.xlsx' file
        ),
    ),
]
