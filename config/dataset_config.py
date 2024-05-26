"""This module contains datasets used in the project. The datasets are defined as dataset dataclasses."""
from dataclasses import dataclass
from config import labels
from masks import masks


@dataclass
class Dataset:
    """This class represents a dataset."""

    uid: str # unique identifier for the dataset
    dataset_name: str # name of the dataset
    phases: dict[str, str] # phase_id used for encoding the phase in img name, phase_name used for naming the folder
    label2radlex: dict[str, list[str]] # some labels have multiple RadLex codes
    color2radlex: dict[int, str] # color that is used to encode source mask


datasets = {
    "KITS23": Dataset(
        uid="00",
        dataset_name="KITS23",
        phases={"0": "CT"}, # arterial or nephrogenic
        label2radlex={"normal": [labels.NormalityDecriptor],
                      "angiomyolipoma": [labels.Angiomyolipoma, labels.Neoplasm],
                      "chromophobe_rcc": [labels.ChromophobeAdenocarcinoma, labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "clear_cell_papillary": [labels.ClearCellAdenocarcinoma, labels.PapillaryRenalAdenocarcinoma, labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "clear_cell_rcc": [labels.ClearCellAdenocarcinoma, labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "mest": [labels.Neoplasm],
                      "multilocular_cystic_rcc": [labels.MultilocularCysticRenalTumor, labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "oncocytoma": [labels.Oncocytoma, labels.Neoplasm],
                      "papillary_rcc": [labels.PapillaryRenalAdenocarcinoma, labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "rcc_unclassified": [labels.RenalAdenocarcinoma, labels.Adenocarcinoma, labels.Neoplasm],
                      "spindle_cell_neoplasm": [labels.Neoplasm],
                      "urothelial": [labels.Neoplasm],
                      "wilms": [labels.WilmsTumor, labels.Neoplasm],
        },
        color2radlex={1: masks.Kidney,
                      2: masks.Neoplasm,
                      3: masks.RenalCyst},
    ),
    "MosMedData": Dataset(
        uid="01",
        dataset_name="MosMedData",
        phases={"0": "HRCT_nocontrast"},
        label2radlex={},
        color2radlex={},
    ),
    "LIDC-IDRI": Dataset(
        uid="02",
        dataset_name="LIDC-IDRI",
        phases={"0": "CT"},
        label2radlex={},
        color2radlex={255: masks.Nodule},
    ),
    "CT_COLONOGRAPHY": Dataset(
        uid="03",
        dataset_name="CT_COLONOGRAPHY",
        phases={"0": "CT"},
        label2radlex={},
        color2radlex={},
    ),
    "Chest_X-ray_Abnormalities_Detection": Dataset(
        uid="04",
        dataset_name="Chest_X-ray_Abnormalities_Detection",
        phases={"0": "Xray"},
        label2radlex={},
        color2radlex={},
    ),
    "CoronaHack_Chest_X-Ray": Dataset(
        uid="05",
        dataset_name="CoronaHack_Chest_X-Ray",
        phases={"0": "Xray"},
        label2radlex={
            "normal": [labels.NormalityDecriptor],
            "PnemoniaBacteria": [labels.Pneumonia],
            "PnemoniaViral": [labels.PneumoniaViral, labels.ViralInfection, labels.Pneumonia],
        },
        color2radlex={},
    ),
    "ChestX-ray14": Dataset(
        uid="06",
        dataset_name="ChestX-ray14",
        phases={"0": "Xray"},
        label2radlex={
            "normal": [labels.NormalityDecriptor],
            "atelectasis": [labels.Atelectasis],
            "cardiomegaly": [labels.BoxlikeHeart],
            "effusion": [labels.PleuralEffusion],
            "infiltration": [labels.Consolidation],
            "mass": [labels.Lesion, labels.Mass],
            "nodule": [labels.Lesion],
            "pneumonia": [labels.Pneumonia],
            "pneumothorax": [labels.Pneumothorax],
            "consolidation": [labels.Consolidation],
            "edema": [labels.PulmonaryEdema],
            "emphysema": [labels.Emphysema],
            "fibrosis": [labels.Fibrosis],
            "pleural_thickening": [labels.Thickening],
            "hernia": [labels.Hernia],
        },
        color2radlex={},
    ),
    "PadChest": Dataset(
        uid="07",
        dataset_name="PadChest",
        phases={"0": "Xray"},
        label2radlex={},
        color2radlex={},
    ),
    "Lung_segmentation_from_Chest_X-Rays": Dataset(
        uid="08",
        dataset_name="Lung_segmentation_from_Chest_X-Rays",
        phases={"0": "Xray"},
        label2radlex={},
        color2radlex={},
    ),
    "Brain_Tumor_Classification_MRI": Dataset(
        uid="09",
        dataset_name="Brain_Tumor_Classification_MRI",
        phases={"0": "T1_weighted_postCM"}, # occasionally T2_weighted! 
        label2radlex={},
        color2radlex={},
    ),
    "Brain_Tumor_Progression": Dataset(
        uid="10",
        dataset_name="Brain_Tumor_Progression",
        phases={"0": "T1_weighted"}, # probably some before CM and some after CM
        label2radlex={},
        color2radlex={},
    ),
    "Qin_Brain_MRI": Dataset(
        uid="11",
        dataset_name="Qin_Brain_MRI",
        phases={"0": "T1_weighted_after_CM",},
        label2radlex={},
        color2radlex={},
    ),
    "BRAIN_MRI_SEGMENTATION": Dataset(
        uid="12",
        dataset_name="BRAIN_MRI_SEGMENTATION",
        phases={"0": "CT",},
        label2radlex={},
        color2radlex={},
    ),
    "CT_ORG": Dataset(
        uid="13",
        dataset_name="CT_ORG",
        phases={"0": "CT",},
        label2radlex={},
        color2radlex={},
    ),
    "StanfordBrainMET": Dataset(
        uid="14",
        dataset_name="StanfordBrainMET",
        phases={
            "0": "T1_weighted_preCM_spin-echo_pre-contrast",
            "1": "T1_weighted_postCM", # This one was used to generate the masks
            "2": "T1_gradient_echo_postCM", # using an IR-prepped FSPGR sequence
            "3": "T2_FLAIR_postCM",
        },
        label2radlex={},
        color2radlex={255: masks.Metastasis},
    ),
    "StanfordCOCA": Dataset(
        uid="15",
        dataset_name="StanfordCOCA",
        phases={"0": "CT"},
        label2radlex={},
        color2radlex={20: masks.CalciumScore},
    ),
    "Alzheimers_Dataset": Dataset(
        uid="16",
        dataset_name="Alzheimers_Dataset",
        phases={"0": "MRI"},
        label2radlex={},
        color2radlex={},
    ),
    "Finding_and_Measuring_Lungs_in_CT_Data": Dataset(
        uid="17",
        dataset_name="Finding_and_Measuring_Lungs_in_CT_Data",
        phases={"0": "CT"},
        label2radlex={},
        color2radlex={255: masks.Lung},
    ),
    "Brain_with_hemorrhage": Dataset(
        uid="19",
        dataset_name="Brain_with_hemorrhage",
        phases={"0": "Bone",
                "1": "Brain"},
        label2radlex={},
        color2radlex={255: masks.hemorrhage},
    ),
    "Brain_Tumor_Detection": Dataset(
        uid="20",
        dataset_name="Brain_Tumor_Detection",
        phases={"0": "MRI"},
        label2radlex={
            "normal": [labels.NormalityDecriptor],
            "glioma": [labels.Glioma],
            "meningioma": [labels.Meningioma],
            "pituitary": [labels.Pituitary],
        },
        color2radlex={},
    ),
    "Covid19_Detection": Dataset(
        uid="22",
        dataset_name="Covid19_Detection",
        phases={"0": "Xray"},
        label2radlex={
            "normal": [labels.NormalityDecriptor],
            "pneumonia_bacterial": [labels.Pneumonia],
            "pneumonia_viral": [labels.PneumoniaViral, labels.Pneumonia, labels.ViralInfection],
        },
        color2radlex={},
    ),
    "Liver_And_Liver_Tumor": Dataset(
        uid="23",
        dataset_name="Liver_And_Liver_Tumor",
        phases={"0": "CT"},
        color2radlex={1: masks.Liver, 1: masks.Neoplasm}, # masks have the same encoding, source keeps them in separate folders to differentiate!
    ),
    "Knee_Osteoarthritis": Dataset(
        uid="24",
        dataset_name="Knee_Osteoarthritis",
        phases={"0": "CT"},
        label2radlex={},
        color2radlex={},
    ),
}
