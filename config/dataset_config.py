"""This module contains datasets used in the project. The datasets are defined as dataset dataclasses."""
from dataclasses import dataclass
from config import labels
from config import masks


@dataclass
class DatasetArgs:
    """This class represents a dataset."""

    uid: str # unique identifier for the dataset
    dataset_name: str # name of the dataset
    phases: dict[str, str] # phase_id used for encoding the phase in img name, phase_name used for naming the folder
    label2radlex: dict[str, list[str]] # some labels have multiple RadLex codes
    source_color2target: dict[int, str] # color that is used to encode source mask


KITS23 = DatasetArgs(
    uid="00",
    dataset_name="KITS23",
    phases={"0": "CT"}, # arterial or nephrogenic
    label2radlex={"normal": [labels.NormalityDecriptor.radlex_name],
                    "angiomyolipoma": [labels.Angiomyolipoma.radlex_name, labels.Neoplasm.radlex_name],
                    "chromophobe_rcc": [labels.ChromophobeAdenocarcinoma.radlex_name, labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "clear_cell_papillary": [labels.ClearCellAdenocarcinoma.radlex_name, labels.PapillaryRenalAdenocarcinoma.radlex_name, labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "clear_cell_rcc": [labels.ClearCellAdenocarcinoma.radlex_name, labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "mest": [labels.Neoplasm.radlex_name],
                    "multilocular_cystic_rcc": [labels.MultilocularCysticRenalTumor.radlex_name, labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "oncocytoma": [labels.Oncocytoma.radlex_name, labels.Neoplasm.radlex_name],
                    "papillary_rcc": [labels.PapillaryRenalAdenocarcinoma.radlex_name, labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "rcc_unclassified": [labels.RenalAdenocarcinoma.radlex_name, labels.Adenocarcinoma.radlex_name, labels.Neoplasm.radlex_name],
                    "spindle_cell_neoplasm": [labels.Neoplasm.radlex_name],
                    "urothelial": [labels.Neoplasm.radlex_name],
                    "wilms": [labels.WilmsTumor.radlex_name, labels.Neoplasm.radlex_name],
    },
    source_color2target={1: masks.Kidney.color,
                  2: masks.Neoplasm.color,
                  3: masks.RenalCyst.color},
)

MosMedData = DatasetArgs(
    uid="01",
    dataset_name="MosMedData",
    phases={"0": "HRCT_nocontrast"},
    label2radlex={},
    source_color2target={},
)

LIDC_IDRI = DatasetArgs(
    uid="02",
    dataset_name="LIDC-IDRI",
    phases={"0": "CT"},
    label2radlex={},
    source_color2target={255: masks.Nodule.color},
)

CT_COLONOGRAPHY = DatasetArgs(
    uid="03",
    dataset_name="CT_COLONOGRAPHY",
    phases={"0": "CT"},
    label2radlex={},
    mask_source_color2target={},
)

Chest_Xray_Abnormalities_Detection = DatasetArgs(
    uid="04",
    dataset_name="Chest_X-ray_Abnormalities_Detection",
    phases={"0": "Xray"},
    label2radlex={},
    source_color2target={},
)

CoronaHack_Chest_XRay = DatasetArgs(
    uid="05",
    dataset_name="CoronaHack_Chest_X-Ray",
    phases={"0": "Xray"},
    label2radlex={
        "normal": [labels.NormalityDecriptor.radlex_name],
        "PnemoniaBacteria": [labels.Pneumonia.radlex_name],
        "PnemoniaViral": [labels.PneumoniaViral.radlex_name, labels.ViralInfection.radlex_name, labels.Pneumonia.radlex_name],
    },
    source_color2target={},
)

ChestXray14 = DatasetArgs(
        uid="06",
        dataset_name="ChestX-ray14",
        phases={"0": "Xray"},
        label2radlex={
            "normal": [labels.NormalityDecriptor.radlex_name],
            "atelectasis": [labels.Atelectasis.radlex_name],
            "cardiomegaly": [labels.BoxlikeHeart.radlex_name],
            "effusion": [labels.PleuralEffusion.radlex_name],
            "infiltration": [labels.Consolidation.radlex_name],
            "mass": [labels.Lesion.radlex_name, labels.Mass.radlex_name],
            "nodule": [labels.Lesion.radlex_name],
            "pneumonia": [labels.Pneumonia.radlex_name],
            "pneumothorax": [labels.Pneumothorax.radlex_name],
            "consolidation": [labels.Consolidation.radlex_name],
            "edema": [labels.PulmonaryEdema.radlex_name],
            "emphysema": [labels.Emphysema.radlex_name],
            "fibrosis": [labels.Fibrosis.radlex_name],
            "pleural_thickening": [labels.Thickening.radlex_name],
            "hernia": [labels.Hernia.radlex_name],
        },
        source_color2target={},
    )

PadChest = DatasetArgs(
    uid="07",
    dataset_name="PadChest",
    phases={"0": "Xray"},
    label2radlex={},
    source_color2target={},
)

Lung_segmentation_from_Chest_XRays = DatasetArgs(
    uid="08",
    dataset_name="Lung_segmentation_from_Chest_X-Rays",
    phases={"0": "Xray"},
    label2radlex={},
    source_color2target={},
)
Brain_Tumor_Classification_MRI = DatasetArgs(
    uid="09",
    dataset_name="Brain_Tumor_Classification_MRI",
    phases={"0": "T1_weighted_postCM"}, # occasionally T2_weighted! 
    label2radlex={},
    source_color2target={},
)

Brain_Tumor_Progression = DatasetArgs(
    uid="10",
    dataset_name="Brain_Tumor_Progression",
    phases={"0": "T1_weighted"}, # probably some before CM and some after CM
    label2radlex={},
    source_color2target={},
)

Qin_Brain_MRI = DatasetArgs(
    uid="11",
    dataset_name="Qin_Brain_MRI",
    phases={"0": "T1_weighted_after_CM",},
    label2radlex={},
    source_color2target={},
)

BRAIN_MRI_SEGMENTATION = DatasetArgs(
    uid="12",
    dataset_name="BRAIN_MRI_SEGMENTATION",
    phases={"0": "CT",},
    label2radlex={},
    source_color2target={},
)

CT_ORG = DatasetArgs(
    uid="13",
    dataset_name="CT_ORG",
    phases={"0": "CT",},
    label2radlex={},
    source_color2target={},
)

StanfordBrainMET = DatasetArgs(
    uid="14",
    dataset_name="StanfordBrainMET",
    phases={
        "0": "T1_weighted_preCM_spin-echo_pre-contrast",
        "1": "T1_weighted_postCM", # This one was used to generate the masks
        "2": "T1_gradient_echo_postCM", # using an IR-prepped FSPGR sequence
        "3": "T2_FLAIR_postCM",
    },
    label2radlex={},
    source_color2target={255: masks.Metastasis.color},
)

StanfordCOCA = DatasetArgs(
    uid="15",
    dataset_name="StanfordCOCA",
    phases={"0": "CT"},
    label2radlex={},
    source_color2target={20: masks.CalciumScore.color},
)

Alzheimers_Dataset = DatasetArgs(
    uid="16",
    dataset_name="Alzheimers_Dataset",
    phases={"0": "MRI"},
    label2radlex={},
    source_color2target={},
)

Finding_and_Measuring_Lungs_in_CT_Data = DatasetArgs(
    uid="17",
    dataset_name="Finding_and_Measuring_Lungs_in_CT_Data",
    phases={"0": "CT"},
    label2radlex={},
    source_color2target={255: masks.Lung.color},
)
    
Brain_with_hemorrhage = DatasetArgs(
    uid="19",
    dataset_name="Brain_with_hemorrhage",
    phases={"0": "Bone",
            "1": "Brain"},
    label2radlex={},
    source_color2target={255: masks.Hemorrhage.color},
)
    
Brain_Tumor_Detection = DatasetArgs(
    uid="20",
    dataset_name="Brain_Tumor_Detection",
    phases={"0": "MRI"},
    label2radlex={
        "normal": [labels.NormalityDecriptor.radlex_name],
        "glioma": [labels.Glioma.radlex_name],
        "meningioma": [labels.Meningioma.radlex_name],
        "pituitary": [labels.Pituitary.radlex_name],
    },
    source_color2target={},
)

Covid19_Detection = DatasetArgs(
    uid="22",
    dataset_name="Covid19_Detection",
    phases={"0": "Xray"},
    label2radlex={
        "normal": [labels.NormalityDecriptor.radlex_name],
        "pneumonia_bacterial": [labels.Pneumonia.radlex_name],
        "pneumonia_viral": [labels.PneumoniaViral.radlex_name, labels.Pneumonia.radlex_name, labels.ViralInfection.radlex_name],
    },
    source_color2target={},
)
   
Liver_And_Liver_Tumor = DatasetArgs(
    uid="23",
    dataset_name="Liver_And_Liver_Tumor",
    phases={"0": "CT"},
    label2radlex={},
    source_color2target={1: masks.Liver.color, 1: masks.Neoplasm.color}, # masks have the same encoding, source keeps them in separate folders to differentiate!
)

Knee_Osteoarthritis = DatasetArgs(
    uid="24",
    dataset_name="Knee_Osteoarthritis",
    phases={"0": "CT"},
    label2radlex={},
    source_color2target={},
)