"""This module contains datasets used in the project. The datasets are defined as dataset dataclasses."""
from dataclasses import dataclass, field

from config import labels, masks


@dataclass
class MaskColor:
    """A class for storing imformation about how to recolor a given mask."""

    source_color: int
    target_color: int


@dataclass
class DatasetArgs:
    """This class represents a dataset."""

    dataset_uid: str  # unique identifier for the dataset
    dataset_name: str  # name of the dataset
    phases: dict[str, str]  # phase_id used for encoding the phase in img name, phase_name used for naming the folder
    labels: dict[str, list[dict[str, float]]] = field(default_factory=dict)  # some labels have multiple RadLex codes
    masks: dict[str, MaskColor] = field(default_factory=dict)  # color that is used to encode source mask


kits23 = DatasetArgs(
    dataset_uid="00",
    dataset_name="kits23",
    phases={"0": "CT"},  # arterial or nephrogenic
    labels={
        "normal": [{labels.NormalityDecriptor.radlex_name: 1}],
        "angiomyolipoma": [{labels.Angiomyolipoma.radlex_name: 1}, {labels.Neoplasm.radlex_name: 1}],
        "chromophobe_rcc": [
            {labels.ChromophobeAdenocarcinoma.radlex_name: 1},
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "clear_cell_papillary": [
            {labels.ClearCellAdenocarcinoma.radlex_name: 1},
            {labels.PapillaryRenalAdenocarcinoma.radlex_name: 1},
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "clear_cell_rcc": [
            {labels.ClearCellAdenocarcinoma.radlex_name: 1},
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "mest": [{labels.Neoplasm.radlex_name: 1}],
        "multilocular_cystic_rcc": [
            {labels.MultilocularCysticRenalTumor.radlex_name: 1},
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "oncocytoma": [{labels.Oncocytoma.radlex_name: 1}, {labels.Neoplasm.radlex_name: 1}],
        "papillary_rcc": [
            {labels.PapillaryRenalAdenocarcinoma.radlex_name: 1},
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "rcc_unclassified": [
            {labels.RenalAdenocarcinoma.radlex_name: 1},
            {labels.Adenocarcinoma.radlex_name: 1},
            {labels.Neoplasm.radlex_name: 1},
        ],
        "spindle_cell_neoplasm": [{labels.Neoplasm.radlex_name: 1}],
        "urothelial": [{labels.Neoplasm.radlex_name: 1}],
        "wilms": [{labels.WilmsTumor.radlex_name: 1}, {labels.Neoplasm.radlex_name: 1}],
    },
    masks={
        masks.Kidney.radlex_name: MaskColor(source_color=1, target_color=masks.Kidney.color),
        masks.Neoplasm.radlex_name: MaskColor(source_color=2, target_color=masks.Neoplasm.color),
        masks.RenalCyst.radlex_name: MaskColor(source_color=3, target_color=masks.RenalCyst.color),
    },
)

coronahack = DatasetArgs(
    dataset_uid="01",
    dataset_name="coronahack",
    phases={"0": "Xray"},
    labels={
        "Normal": [{labels.NormalityDecriptor.radlex_name: 1}],
        "PneumoniaBacteria": [{labels.Pneumonia.radlex_name: 1}],
        "PneumoniaVirus": [
            {labels.PneumoniaViral.radlex_name: 1},
            {labels.ViralInfection.radlex_name: 1},
            {labels.Pneumonia.radlex_name: 1},
        ],
    },
)

alzheimers = DatasetArgs(
    dataset_uid="02",
    dataset_name="alzheimers",
    phases={"0": "MRI"},
    labels={
        "MildDemented": [{labels.Dementia.radlex_name: 2 / 3}],
        "ModerateDemented": [{labels.Dementia.radlex_name: 1}],
        "NonDemented": [{labels.NormalityDecriptor.radlex_name: 1}],
        "VeryMildDemented": [{labels.Dementia.radlex_name: 1 / 3}],
    },
)

chest_xray_masks_and_labels = DatasetArgs(
    dataset_uid="03",
    dataset_name="chest_xray_masks_and_labels",
    phases={"0": "Xray"},
)

brain_tumor_classification = DatasetArgs(
    dataset_uid="04",
    dataset_name="brain_tumor_classification",
    phases={"0": "T1_weighted_postCM"},  # occasionally T2_weighted!
)

covid19_detection = DatasetArgs(
    dataset_uid="05",
    dataset_name="covid19_detection",
    phases={"0": "Xray"},
    labels={
        "Normal": [{labels.NormalityDecriptor.radlex_name: 1}],
        "BacterialPneumonia": [{labels.Pneumonia.radlex_name: 1}],
        "ViralPneumonia": [
            {labels.PneumoniaViral.radlex_name: 1},
            {labels.Pneumonia.radlex_name: 1},
            {labels.ViralInfection.radlex_name: 1},
        ],
        "COVID-19": [
            {labels.PneumoniaViral.radlex_name: 1},
            {labels.Pneumonia.radlex_name: 1},
            {labels.ViralInfection.radlex_name: 1},
        ],
    },
)

finding_and_measuring_lungs = DatasetArgs(
    dataset_uid="06",
    dataset_name="finding_and_measuring_lungs",
    phases={"0": "CT"},
    masks={masks.Lung.radlex_name: MaskColor(source_color=255, target_color=masks.Lung.color)},
)

brain_with_intracranial_hemorrhage = DatasetArgs(
    dataset_uid="07",
    dataset_name="brain_with_intracranial_hemorrhage",
    phases={"0": "Bone", "1": "Brain"},
    labels={
        "brain_hemorrhage": [{labels.Hemorrhage.radlex_name: 1}],
        "normal": [{labels.NormalityDecriptor.radlex_name: 1}],
    },
    masks={masks.Hemorrhage.radlex_name: MaskColor(source_color=255, target_color=masks.Hemorrhage.color)},
)

lits = DatasetArgs(
    dataset_uid="08",
    dataset_name="lits",
    phases={"0": "CT"},
    labels={
        "Neoplasm": [{labels.Neoplasm.radlex_name: 1}],
        "NormalityDescriptor": [{labels.NormalityDecriptor.radlex_name: 1}],
    },
    masks={
        masks.Liver.radlex_name: MaskColor(source_color=1, target_color=masks.Liver.color),
        masks.Neoplasm.radlex_name: MaskColor(source_color=1, target_color=masks.Neoplasm.color),
    },  # masks have the same encoding, source keeps them in separate folders to differentiate!
)

brain_tumor_detection = DatasetArgs(
    dataset_uid="09",
    dataset_name="brain_tumor_detection",
    phases={"0": "MRI"},
    labels={
        "Y": [{labels.NormalityDecriptor.radlex_name: 1}],
        "N": [{labels.Neoplasm.radlex_name: 1}],
    },
)

knee_osteoarthritis = DatasetArgs(
    dataset_uid="10",
    dataset_name="knee_osteoarthritis",
    phases={"0": "CT"},
    labels={
        "0": [{labels.NormalityDecriptor.radlex_name: 1}],
        "1": [{labels.Osteoarthritis.radlex_name: 0.25}],
        "2": [{labels.Osteoarthritis.radlex_name: 0.5}],
        "3": [{labels.Osteoarthritis.radlex_name: 0.75}],
        "4": [{labels.Osteoarthritis.radlex_name: 1}],
    },
)

qin_brain_dsc_mri = DatasetArgs(
    dataset_uid="11",
    dataset_name="qin_brain_dsc_mri",
    phases={
        "0": "T1_weighted_after_CM",
    },
    labels={
        "normal": [{labels.NormalityDecriptor.radlex_name: 1}],
        "glioma": [{labels.Glioma.radlex_name: 1}],
        "meningioma": [{labels.Meningioma.radlex_name: 1}],
        "pituitary": [{labels.Pituitary.radlex_name: 1}],
    },
    masks={},
)

lidc_idri = DatasetArgs(
    dataset_uid="12",
    dataset_name="lidc_idri",
    phases={"0": "CT"},
    masks={masks.Nodule.radlex_name: MaskColor(source_color=255, target_color=masks.Nodule.color)},
)

ct_org = DatasetArgs(
    dataset_uid="13",
    dataset_name="ct_org",
    phases={
        "0": "CT",
    },
)

brain_tumor_progression = DatasetArgs(
    dataset_uid="14",
    dataset_name="brain_tumor_progression",
    phases={"0": "T1_weighted"},  # probably some before CM and some after CM
    labels={},
    masks={},
)

chest_xray14 = DatasetArgs(
    dataset_uid="15",
    dataset_name="chest_xray14",
    phases={"0": "Xray"},
    labels={
        "No Finding": [{labels.NormalityDecriptor.radlex_name: 1}],
        "Atelectasis": [{labels.Atelectasis.radlex_name: 1}],
        "Cardiomegaly": [{labels.BoxlikeHeart.radlex_name: 1}],
        "Effusion": [{labels.PleuralEffusion.radlex_name: 1}],
        "Infiltration": [{labels.Consolidation.radlex_name: 1}],
        "Mass": [{labels.Lesion.radlex_name: 1}, {labels.Mass.radlex_name: 1}],
        "Nodule": [{labels.Lesion.radlex_name: 1}],
        "Pneumonia": [{labels.Pneumonia.radlex_name: 1}],
        "Pneumothorax": [{labels.Pneumothorax.radlex_name: 1}],
        "Consolidation": [{labels.Consolidation.radlex_name: 1}],
        "Edema": [{labels.PulmonaryEdema.radlex_name: 1}],
        "Emphysema": [{labels.Emphysema.radlex_name: 1}],
        "Fibrosis": [{labels.Fibrosis.radlex_name: 1}],
        "Pleural_Thickening": [{labels.Thickening.radlex_name: 1}],
        "Hernia": [{labels.Hernia.radlex_name: 1}],
    },
)

brain_met_share = DatasetArgs(
    dataset_uid="16",
    dataset_name="brain_met_share",
    phases={
        "0": "T1_weighted_preCM_spin-echo_pre-contrast",
        "1": "T1_weighted_postCM",  # This one was used to generate the masks
        "2": "T1_gradient_echo_postCM",  # using an IR-prepped FSPGR sequence
        "3": "T2_FLAIR_postCM",
    },
    masks={masks.Metastasis.radlex_name: MaskColor(source_color=255, target_color=masks.Metastasis.color)},
)

coca = DatasetArgs(
    dataset_uid="17",
    dataset_name="coca",
    phases={"0": "CT"},
    masks={masks.CalciumScore.radlex_name: MaskColor(source_color=20, target_color=masks.CalciumScore.color)},
)
