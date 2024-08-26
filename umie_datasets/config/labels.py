"""
All target labels used in UMIE.

Each label has a unique id, RadLex name, RadLex id, grade, and source names.
Source names are the names of the labels used in the original datasets. They are meant to be helpful when translating labels from new source datasets to RadLex labels.
Each of the target labels has a grade, which is an integer value that represents the intensity of the label ranging from 0 to 1.
E.g. 0.25 - mild, 0.5 - moderate, 0.75 - severe, 1 - critical. If the grade is not specified, the default value is 1.
Here in "grades", we store the information about how many grades the label has.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Label:  # Name of the label should match RadLex name
    """This class represents a target UMIE label."""

    id: int  # Unique id of the label in UMIE
    radlex_name: str  # name of the label in RadLex
    radlex_id: str  # unique id of the label in RadLex
    grades: Optional[int] = field(  # number of grades the label has (possible intensity levels of the label)
        default_factory=int
    )  # Grades of the label, some datasets come with intesity scale for the labels
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # Names used by other datasets, key - dataset name, value - source label name in this dataset


# Use camel case for the labels names
# We do not keep vague labels like "other" or "unknown"
# One source label may correspond to multiple RadLex labels

NormalityDecriptor = Label(
    id=0,
    radlex_name="NormalityDecriptor",
    radlex_id="RID29001",
    source_names={
        "kits23": ["normal"],
        "coronahack": ["Normal"],
        "alzheimers": ["NonDemented"],
        "covid19_detection": ["normal"],
        "chestX-ray14": ["No Finding"],
        "brain_tumor_classification": ["no_tumor"],
        "brain_with_intracranial_hemorrhage": ["normal"],
        "brain_tumor_detection": ["N"],
        "lits": ["NormalityDescriptor"],
    },
)

Neoplasm = Label(
    id=1,
    radlex_name="Neoplasm",
    radlex_id="RID3957",
    source_names={
        "kits23": [
            "angiomyolipoma",
            "chromophobe",
            "clear_cell_papillary_rcc",
            "clear_cell_rcc",
            "mest",
            "multilocular_cystic_rcc",
            "oncocytoma",
            "papillary_rcc",
            "rcc_unclassified",
            "spindle_cell_neoplasm",
            "urothelial",
            "wilms",
            "other",
        ],
        "pad_chest": ["adenocarcinoma"],
        "brain_tumor_detection": ["Y"],
        "lits": ["Neoplasm"],
    },
)

RenalAdenocarcinoma = Label(
    id=2,
    radlex_name="RenalAdenocarcinoma",
    radlex_id="RID4234",
    source_names={
        "kits23": [
            "clear_cell_rcc",
            "chromophobe_rcc",
            "papillary_rcc",
            "multilocular_cystic_rcc",
            "rcc_unclassified",
            "clear_cell_papillary",
        ],
    },
)

ClearCellAdenocarcinoma = Label(
    id=3,
    radlex_name="ClearCellAdenocarcinoma",
    radlex_id="RID4235",
    source_names={"kits23": ["clear_cell_rcc", "clear_cell_papillary"]},
)

ChromophobeAdenocarcinoma = Label(
    id=4, radlex_name="ChromophobeAdenocarcinoma", radlex_id="RID4236", source_names={"kits23": ["chromophobe_rcc"]}
)

TransitionalCellCarcinoma = Label(
    id=5,
    radlex_name="TransitionalCellCarcinoma",
    radlex_id="",
    source_names={"kits23": ["transitional_cell_carcinoma"]},
)

PapillaryRenalAdenocarcinoma = Label(
    id=6,
    radlex_name="PapillaryRenalAdenocarcinoma",
    radlex_id="RID4233",
    source_names={"kits23": ["papillary_rcc", "clear_cell_papillary"]},
)

MultilocularCysticRenalTumor = Label(
    id=7,
    radlex_name="MultilocularCysticRenalTumor",
    radlex_id="RID4538",
    source_names={"kits23": ["multilocular_cystic_rcc"]},
)

WilmsTumor = Label(id=8, radlex_name="WilmsTumor", radlex_id="RID4553", source_names={"kits23": ["wilms"]})

Angiomyolipoma = Label(
    id=9, radlex_name="Angiomyolipoma", radlex_id="RID4343", source_names={"kits23": ["angiomyolipoma"]}
)

Oncocytoma = Label(id=10, radlex_name="Oncocytoma", radlex_id="RID4515", source_names={"kits23": ["oncocytoma"]})

RenalCyst = Label(id=11, radlex_name="RenalCyst", radlex_id="RID35811", source_names={"kits23": ["cyst"]})

ViralInfection = Label(
    id=12,
    radlex_name="ViralInfection",
    radlex_id="RID4687",
    grades=5,
    source_names={
        "coronahack": ["PneumoniaVirus"],
        "MosMedData": ["CT-1", "CT-2", "CT-3", "CT-4"],
        "covid19_detection": ["pneumonia_viral"],
    },
)

Pneumonia = Label(
    id=13,
    radlex_name="Pneumonia",
    radlex_id="RID5350",
    source_names={
        "coronahack": ["PneumoniaVirus", "PneumoniaBacteria"],
        "ChestX-ray14": ["Pneumonia"],
        "PadChest": ["Pneumonia", "atypical pneumonia"],
        "covid19_detection": ["pneumonia_bacterial", "pneumonia_viral"],
    },
)

PneumoniaViral = Label(
    id=14,
    radlex_name="PneumoniaViral",
    radlex_id="RID34769",
    source_names={"coronahack": ["PneumoniaVirus"], "covid19_detection": ["pneumonia_viral"]},
)

Atelectasis = Label(
    id=15,
    radlex_name="Atelectasis",
    radlex_id="RID28493",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Atelectasis"], "ChestX-ray14": ["Atelectasis"]},
)

Calcification = Label(
    id=16,
    radlex_name="Calcification",
    radlex_id="RID5196",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Calcification"]},
)

BoxlikeHeart = Label(
    id=17,
    radlex_name="BoxlikeHeart",
    radlex_id="RID35057",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Cardiomegaly"], "ChestX-ray14": ["Cardiomegaly"]},
)

Consolidation = Label(
    id=18,
    radlex_name="Consolidation",
    radlex_id="RID43255",
    source_names={
        "Chest_X-ray_Abnormalities_Detection": ["Consolidation", "Infiltration"],
        "ChestX-ray14": ["Consolidation", "Infiltration"],
    },
)

InterstitialLungDisease = Label(
    id=19,
    radlex_name="InterstitialLungDisease",
    radlex_id="RID28799",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["ILD"]},
)

Opacity = Label(
    id=20,
    radlex_name="Opacity",
    radlex_id="RID28530",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Opacity"]},
)

Lesion = Label(
    id=21,
    radlex_name="Lesion",
    radlex_id="RID38780",
    source_names={
        "Chest_X-ray_Abnormalities_Detection": ["Nodule/Mass", "Other Lesion"],
        "ChestX-ray14": ["Mass", "Nodule"],
    },
)

PleuralEffusion = Label(
    id=22,
    radlex_name="PleuralEffusion",
    radlex_id="RID34539",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Effusion"], "ChestX-ray14": ["Effusion"]},
)

Thickening = Label(
    id=23,
    radlex_name="Thickening",
    radlex_id="RID5352",
    source_names={
        "Chest_X-ray_Abnormalities_Detection": ["Pleural_Thickening"],
        "ChestX-ray14": ["Pleural_Thickening"],
    },
)

Pneumothorax = Label(
    id=24,
    radlex_name="Pneumothorax",
    radlex_id="RID5352",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Pneumothorax"], "ChestX-ray14": ["Pneumothorax"]},
)

Fibrosis = Label(
    id=25,
    radlex_name="Fibrosis",
    radlex_id="RID3820",
    source_names={
        "Chest_X-ray_Abnormalities_Detection": ["Pulmonary Fibrosis"],
        "ChestX-ray14": ["Fibrosis"],
        "PadChest": ["Pulmonary Fibrosis"],
    },
)

Mass = Label(id=26, radlex_name="Mass", radlex_id="RID3874", source_names={"ChestX-ray14": ["Mass"]})

PulmonaryEdema = Label(
    id=27,
    radlex_name="PulmonaryEdema",
    radlex_id="RID4866",
    source_names={"ChestX-ray14": ["Edema"], "PadChest": ["Pulmonary Edema"]},
)

Emphysema = Label(
    id=28,
    radlex_name="Emphysema",
    radlex_id="RID4799",
    source_names={"ChestX-ray14": ["Emphysema"], "PadChest": ["Emphysema"]},
)

Hernia = Label(id=29, radlex_name="Hernia", radlex_id="RID4895", source_names={"ChestX-ray14": ["Hernia"]})

ChronicObstructivePulmonaryDisease = Label(
    id=30,
    radlex_name="ChronicObstructivePulmonaryDisease",
    radlex_id="RID5317",
    source_names={"PadChest": ["COPD signs"]},
)

Tubeculosis = Label(
    id=31,
    radlex_name="Tubeculosis",
    radlex_id="RID29116",
    source_names={"PadChest": ["Tuberculosis", "Tuberculosis seqelae"]},
)

Metastasis = Label(
    id=32,
    radlex_name="Metastasis",
    radlex_id="RID5231",
    source_names={"PadChest": ["Lung metastasis", "Bone metastasis"]},
)

Pneumonitis = Label(
    id=33, radlex_name="Pneumonitis", radlex_id="RID3541", source_names={"PadChest": ["post radiotherapy changes"]}
)

PulmonaryHypertension = Label(
    id=34,
    radlex_name="PulmonaryHypertension",
    radlex_id="RID3299",
    source_names={"PadChest": ["Pulmonary artery hypertension"]},
)

AdultRespiratoryDistressSyndrome = Label(
    id=35,
    radlex_name="AdultRespiratoryDistressSyndrome",
    radlex_id="RID5319",
    source_names={"PadChest": ["Respiratory distress syndrome"]},
)

Asbestosis = Label(
    id=36, radlex_name="Asbestosis", radlex_id="RID5346", source_names={"PadChest": {"Asbestosis signs"}}
)

Carcinomatosis = Label(
    id=37, radlex_name="Carcinomatosis", radlex_id="RID5231", source_names={"PadChest": ["lymphangitis carcinomatosa"]}
)

Adenocarcinoma = Label(
    id=38,
    radlex_name="Adenocarcinoma",
    radlex_id="RID4226",
    source_names={
        "kits23": [
            "clear_cell_rcc",
            "chromophobe_rcc",
            "papillary_rcc",
            "multilocular_cystic_rcc",
            "rcc_unclassified",
            "clear_cell_papillary",
        ],
        "PadChest": ["adenocarcinoma"],
    },
)

Glioma = Label(
    id=39, radlex_name="Glioma", radlex_id="RID4026", source_names={"brain_tumor_classification": ["glioma_tumor"]}
)

Meningioma = Label(
    id=40,
    radlex_name="Meningioma",
    radlex_id="RID4088",
    source_names={"brain_tumor_classification": ["meningioma_tumor"]},
)

Pituitary = Label(
    id=41,
    radlex_name="Pituitary",
    radlex_id="RID28679",
    source_names={"brain_tumor_classification": ["pituitary_tumor"]},
)

Osteoarthritis = Label(
    id=42,
    radlex_name="Osteoarthritis",
    radlex_id="RID3555",
    grades=4,
    source_names={
        "knee_osteoarthritis": [
            "DoubtfulOsteoarthritis",
            "MinimalOsteoarthritis",
            "ModerateOsteoarthritis",
            "SevereOsteoarthritis",
        ]
    },
)

Hemorrhage = Label(
    id=43,
    radlex_name="Hemorrhage",
    radlex_id="RID4700",
    source_names={"Brain_with_hemorrhage": ["brain_hemorrhage"]},
)

Dementia = Label(
    id=44,
    radlex_name="Dementia",
    radlex_id="RID5136",
    grades=3,
    source_names={"alzheimers": ["VeryMildDemented", "MildDemented", "ModerateDemented"]},
)

HeartFailure = Label(
    id=45,
    radlex_name="HeartFailure",
    radlex_id="RID34795",
    source_names={"PadChest": ["heart insufficiency"]},
)

all_labels = [obj for name, obj in globals().items() if isinstance(obj, Label)]
