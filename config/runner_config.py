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
            source_path="../datasets/kits23/dataset",  # Path to the "dataset" directory in KITS23 repo
            masks_path="../datasets/kits23/dataset",  # Path to the "dataset" directory in KITS23 repo
            target_path=TARGET_PATH,
            labels_path="../datasets/kits23/dataset/kits23.json",  # Path to kits23.json
        ),
    ),
    CoronaHackPipeline(
        path_args=PathArgs(
            source_path="../datasets/praveengovi/coronahack-chest-xraydataset/versions/3",
            target_path=TARGET_PATH,
            labels_path="./datasets/praveengovi/coronahack-chest-xraydataset/versions/3/Chest_xray_Corona_Metadata.csv",  # Path to Chest_xray_Corona_Metadata.csv
        ),
    ),
    AlzheimersPipeline(
        path_args=PathArgs(
            source_path=".../datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images/versions/1",  # Path to archive directory from kaggle
            target_path=TARGET_PATH,
        ),
    ),
    BrainTumorClassificationPipeline(
        path_args=PathArgs(
            source_path="../datasets/nikhilpandey360/chest-xray-masks-and-labels/versions/1",
            target_path=TARGET_PATH,
        ),
    ),
    COVID19DetectionPipeline(
        path_args=PathArgs(
            source_path="../datasets/darshan1504/covid19-detection-xray-dataset/versions/1",
            target_path=TARGET_PATH,
        ),
    ),
    FindingAndMeasuringLungsPipeline(
        path_args=PathArgs(
            source_path="../datasets/kmader/finding-lungs-in-ct-data/versions/2",  # Path to 2d_images directory
            target_path=TARGET_PATH,
            masks_path="",  # Path to 2d_masks directory
        ),
    ),
    BrainWithIntracranialHemorrhagePipeline(
        path_args=PathArgs(
            source_path="../datasets/vbookshelf/computed-tomography-ct-images/versions/1",
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
            source_path="./datasets/jjprotube/brain-mri-images-for-brain-tumor-detection/versions/1",
            target_path=TARGET_PATH,
        ),
    ),
    KneeOsteoarthritisPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
        ),
    ),
    BrainTumorProgressionPipeline(
        path_args=PathArgs(
            source_path="",
            target_path=TARGET_PATH,
            masks_path="",
        )
    ),
    ChestXray14Pipeline(
        path_args=PathArgs(
            source_path="",  # path to images/
            target_path=TARGET_PATH,
            labels_path="",  # Path to Data_Entry_2017_v2020.csv
        ),
    ),
    COCAPipeline(
        path_args=PathArgs(
            source_path="",  # Path to Gated_release_final/patient
            target_path=TARGET_PATH,
            masks_path="",  # Path to Gated_release_final/calcium_xml
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
            source_path="",
            target_path=TARGET_PATH,
            masks_path="",
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
            source_path="",  # Path to 'manifest-{xxxxxxxxxxxxx}/CMMD' folder
            target_path=TARGET_PATH,
            labels_path="",  # Path to 'CMMD_clinicaldata_revision.xlsx' file
        ),
    ),
]
