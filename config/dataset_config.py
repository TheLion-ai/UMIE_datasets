"""This module contains datasets used in the project. The datasets are defined as dataset dataclasses."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Dataset:
    """This class represents a dataset."""

    uid: str
    dataset_name: str
    phases: dict[str, str]
    label2radlex: Optional[dict] = field(default_factory=dict)
    color2mask: Optional[dict] = field(default_factory=dict)
    masks: Optional[list] = field(default_factory=list)


datasets = {
    "KITS23": Dataset(
        uid="00",
        dataset_name="KITS23",
        phases={"0": "CT_arterial"},
        label2radlex={
            "normal": "RID29001",
            "angiomyolipoma": "RID4343",
            "chromophobe": "RID4236",
            "clear_cell_papillary_rcc": "RID4233",
            "clear_cell_rcc": "RID4249",
            "mest": "",
            "multilocular_cystic_rcc": "RID4538",
            "oncocytoma": "RID4515",
            "papillary": "",
            "rcc_unclassified": "",
            "spindle_cell_neoplasm": "",
            "urothelial": "",
            "wilms": "RID4553",
        },
        color2mask={1: "kidney", 2: "kidney_tumor", 3: "kidney_cyst"},
        masks=["kidney", "tumor", "kidney_cyst"],
    ),
    "LIDC-IDRI": Dataset(
        uid="02",
        dataset_name="LIDC-IDRI",
        phases={"0": "CT"},
        color2mask={255: "nodule"},
        masks=["nodule"],
    ),
    "CoronaHack_Chest_X-Ray": Dataset(
        uid="05",
        dataset_name="CoronaHack_Chest_X-Ray",
        phases={"0": "Xray"},
        label2radlex={
            "normal": "",
            "pneumonia_bacterial": "",
            "pneumonia_viral": "RID34769",
        },
    ),
    "ChestX-ray14": Dataset(
        uid="06",
        dataset_name="ChestX-ray14",
        phases={"0": "Xray"},
        label2radlex={
            "normal": "",
            "atelectasis": "RID103",
            "cardiomegaly": "RID104",
            "effusion": "RID105",
            "infiltration": "RID106",
            "mass": "RID107",
            "nodule": "RID108",
            "pneumonia": "RID109",
            "pneumothorax": "RID110",
            "consolidation": "RID111",
            "edema": "RID112",
            "emphysema": "RID113",
            "fibrosis": "RID114",
            "pleural_thickening": "RID115",
            "hernia": "RID116",
        },
    ),
    "StanfordBrainMET": Dataset(
        uid="14",
        dataset_name="StanfordBrainMET",
        phases={
            "0": "T1_weighted_preCM_spin-echo_pre-contrast",
            "1": "T1_weighted_postCM",
            "2": "T1_gradient_echo_postCM",
            "3": "T2_FLAIR_postCM",
        },
        color2mask={255: "metastasis"},
        masks=["metastasis"],
    ),
    "StanfordCOCA": Dataset(
        uid="15",
        dataset_name="StanfordCOCA",
        phases={"0": "CT"},
        color2mask={20: "coronary_artery_calcium"},
        masks=["coronary_artery_calcium"],
    ),
    "Alzheimers_Dataset": Dataset(
        uid="16",
        dataset_name="Alzheimers_Dataset",
        phases={"0": "MRI"},
    ),
    "Finding_and_Measuring_Lungs_in_CT_Data": Dataset(
        uid="17",
        dataset_name="Finding_and_Measuring_Lungs_in_CT_Data",
        phases={"0": "CT"},
        color2mask={255: "lungs"},
        masks=["lungs"],
    ),
    "Brain_with_hemorrhage": Dataset(
        uid="19",
        dataset_name="Brain_with_hemorrhage",
        phases={"0": "CT"},
        color2mask={255: "hemorrhage"},
        masks=["hemorrhage"],
    ),
    "Brain_Tumor_Detection": Dataset(
        uid="20",
        dataset_name="Brain_Tumor_Detection",
        phases={"0": "MRI"},
        label2radlex={
            "normal": "",
            "tumor": "RID113",
            "glioma": "RID114",
            "meningioma": "RID115",
            "pituitary": "RID116",
        },
    ),
    "Covid19_Detection": Dataset(
        uid="22",
        dataset_name="Covid19_Detection",
        phases={"0": "Xray"},
        label2radlex={
            "normal": "",
            "pneumonia_bacterial": "",
            "pneumonia_viral": "RID34769",
        },
    ),
}
