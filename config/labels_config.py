"""Labels fromm all of the datasets."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Label:  # Name of the label should match RadLex name
    """This class represents a label."""

    id: int
    radlex_id: str
    grades: Optional[int] = field(
        default_factory=int
    )  # Grades of the label, some datasets come with intesity scale for the labels
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # Names used by other datasets, key - dataset name, value - label name


# Use camel case for the labels keys
# We do not keep vague labels like "other" or "unknown"
# One source label may correspond to multiple RadLex labels
labels = {
    "NormalityDecriptor": Label(
        id=0,
        radlex_id="RID29001",
        source_names={"ChestX-ray14": ["No Finding"], "Brain_Tumor_Classification_MRI": ["no tumor"]},
    ),
    "Neoplasm": Label(
        id=1,
        radlex_id="RID3957",
        source_names={
            "KITS23": [
                "angiomyolipoma",
                "chromophobe",
                "clear_cell_papillary_rcc",
                "clear_cell_rcc",
                "mest",
                "multilocular_cystic_rcc",
                "oncocytoma",
                "papillary",
                "rcc_unclassified",
                "spindle_cell_neoplasm",
                "urothelial",
                "wilms",
            ],
            "PadChest": ["adenocarcinoma"],
            "Brain_Tumor_Detection": ["tumor"],
        },
    ),
    "Renal_Adenocarcinoma": Label(
        id=2,
        radlex_id="RID4234",
        source_names={
            "KITS23": [
                "clear_cell_rcc",
                "chromophobe_rcc",
                "papillary_rcc",
                "multilocular_cystic_rcc" "rcc_unclassified",
                "clear_cell_papillary",
            ],
        },
    ),
    "ClearCellAdenocarcinoma": Label(
        id=3, radlex_id="RID4235", source_names={"KITS23": ["clear_cell_rcc", "clear_cell_papillary"]}
    ),
    "ChromophobeAdenocarcinoma": Label(id=4, radlex_id="RID4236", source_names={"KITS23": ["chromophobe_rcc"]}),
    "TransitionalCellCarcinoma": Label(id=5, radlex_id="", source_names={"KITS23": ["transitional_cell_carcinoma"]}),
    "PapillaryRenalAdenocarcinoma": Label(
        id=6, radlex_id="RID4233", source_names={"KITS23": ["papillary_rcc", "clear_cell_papillary"]}
    ),
    "MultilocularCysticRenalTumor": Label(
        id=7, radlex_id="RID4538", source_names={"KITS23": ["multilocular_cystic_rcc"]}
    ),
    "WilmsTumor": Label(id=8, radlex_id="RID4553", source_names={"KITS23": ["wilms"]}),
    "Angiomyolipoma": Label(id=9, radlex_id="RID4343", source_names={"KITS23": ["angiomyolipoma"]}),
    "Oncocytoma": Label(id=10, radlex_id="RID4515", source_names={"KITS23": ["oncocytoma"]}),
    "RenalCyst": Label(id=11, radlex_id="RID35811", source_names={"KITS23": ["cyst"]}),
    "ViralInfection": Label(
        id=12,
        radlex_id="RID4687",
        grades=5,
        source_names={
            "CoronaHack_Chest_X-Ray_Dataset": ["PneumoniaViral"],
            "MosMedData": ["CT-1", "CT-2", "CT-3", "CT-4"],
        },
    ),
    "Pneumonia": Label(
        id=13,
        radlex_id="RID5350",
        source_names={
            "CoronaHack_Chest_X-Ray_Dataset": ["PneumoniaViral", "PneumoniaBacteria"],
            "ChestX-ray14": ["Pneumonia"],
            "PadChest": ["Pneumonia", "atypical pneumonia"],
        },
    ),
    "PneumoniaViral": Label(
        id=14, radlex_id="RID34769", source_names={"CoronaHack_Chest_X-Ray_Dataset": ["PneumoniaViral"]}
    ),
    "Atelectasis": Label(
        id=15,
        radlex_id="RID28493",
        source_names={"Chest_X-ray_Abnormalities_Detection": ["Atelectasis"], "ChestX-ray14": ["Atelectasis"]},
    ),
    "Calcification": Label(
        id=16, radlex_id="RID5196", source_names={"Chest_X-ray_Abnormalities_Detection": ["Calcification"]}
    ),
    "Boxlike_Heart": Label(
        id=17,
        radlex_id="RID35057",
        source_names={"Chest_X-ray_Abnormalities_Detection": ["Cardiomegaly"], "ChestX-ray14": ["Cardiomegaly"]},
    ),
    "Consolidation": Label(
        id=18,
        radlex_id="RID43255",
        source_names={
            "Chest_X-ray_Abnormalities_Detection": ["Consolidation", "Infiltration"],
            "ChestX-ray14": ["Consolidation", "Infiltration"],
        },
    ),
    "InterstitialLungDisease": Label(
        id=19, radlex_id="RID28799", source_names={"Chest_X-ray_Abnormalities_Detection": ["ILD"]}
    ),
    "Opacity": Label(id=20, radlex_id="RID28530", source_names={"Chest_X-ray_Abnormalities_Detection": ["Opacity"]}),
    "Lesion": Label(
        id=21,
        radlex_id="",
        source_names={
            "Chest_X-ray_Abnormalities_Detection": ["Nodule/Mass", "Other Lesion"],
            "ChestX-ray14": ["Mass", "Nodule"],
        },
    ),
    "PleuralEffusion": Label(
        id=22,
        radlex_id="RID34539",
        source_names={"Chest_X-ray_Abnormalities_Detection": ["Effusion"], "ChestX-ray14": ["Effusion"]},
    ),
    "Thickening": Label(
        id=23,
        radlex_id="RID5352",
        source_names={
            "Chest_X-ray_Abnormalities_Detection": ["Pleural_Thickening"],
            "ChestX-ray14": ["Pleural_Thickening"],
        },
    ),
    "Pneumothorax": Label(
        id=24,
        radlex_id="RID5352",
        source_names={"Chest_X-ray_Abnormalities_Detection": ["Pneumothorax"], "ChestX-ray14": ["Pneumothorax"]},
    ),
    "Fibrosis": Label(
        id=25,
        radlex_id="RID3820",
        source_names={
            "Chest_X-ray_Abnormalities_Detection": ["Pulmonary Fibrosis"],
            "ChestX-ray14": ["Fibrosis"],
            "PadChest": ["Pulmonary Fibrosis"],
        },
    ),
    "Mass": Label(id=26, radlex_id="RID3874", source_names={"ChestX-ray14": ["Mass"]}),
    "PulmonaryEdema": Label(
        id=27,
        radlex_id="RID4866",
        source_names={"ChestX-ray14": ["Edema"], "PadChest": ["Pulmonary Edema"]},
    ),
    "Emphysema": Label(
        id=28, radlex_id="RID4799", source_names={"ChestX-ray14": ["Emphysema"], "PadChest": ["Emphysema"]}
    ),
    "Heart": Label(id=29, radlex_id="RID4895", source_names={"ChestX-ray14": ["Hernia"]}),
    "ChronicObstructivePulmonaryDisease": Label(id=30, radlex_id="RID5317", source_names={"PadChest": ["COPD signs"]}),
    "Tubeculosis": Label(
        id=31, radlex_id="RID29116", source_names={"PadChest": ["Tuberculosis", "Tuberculosis seqelae"]}
    ),
    "Metastasis": Label(id=32, radlex_id="RID5231", source_names={"PadChest": ["Lung metastasis", "Bone metastasis"]}),
    "Pneumonitis": Label(id=33, radlex_id="RID3541", source_names={"PadChest": ["post radiotherapy changes"]}),
    "PulmonaryHypertension": Label(
        id=34, radlex_id="RID3299", source_names={"PadChest": ["Pulmonary artery hypertension"]}
    ),
    "AdultRespiratoryDistressSyndrome": Label(
        id=35, radlex_id="RID5319", source_names={"PadChest": ["Respiratory distress syndrome"]}
    ),
    "Asbestosis": Label(id=36, radlex_id="RID5346", source_names={"PadChest": {"Asbestosis signs"}}),
    "Carcinomatosis": Label(id=37, radlex_id="RID5231", source_names={"PadChest": ["lymphangitis carcinomatosa"]}),
    "Adenocarcinoma": Label(
        id=38,
        radlex_id="RID4226",
        source_names={
            "KITS23": [
                "clear_cell_rcc",
                "chromophobe_rcc",
                "papillary_rcc",
                "multilocular_cystic_rcc" "rcc_unclassified",
                "clear_cell_papillary",
            ],
            "PadChest": ["adenocarcinoma"],
        },
    ),
    "Glioma": Label(id=39, radlex_id="RID4026", source_names={"Brain_Tumor_Classification_MRI": ["glioma tumor"]}),
    "Meningioma": Label(
        id=40, radlex_id="RID4088", source_names={"Brain_Tumor_Classification_MRI": ["meningioma tumor"]}
    ),
    "Pituitary": Label(
        id=41, radlex_id="RID28679", source_names={"Brain_Tumor_Classification_MRI": ["pituitary tumor"]}
    ),
    "Osteoarthritis": Label(
        id=42,
        radlex_id="RID3555",
        grades=6,
        source_names={"Brain_Tumor_Classification_MRI": ["Healthy",
                                                         "Large osteophytes"]},
    ),
    "Hemorrhage": Label(id=43, radlex_id="RID4700", source_names={"Brain_with_hemorrhage": ["hemorrhage"]}),
    "Dementia": Label(
        id=44,
        radlex_id="RID5136",
        source_names={"Alzheimers_Dataset": ["VeryMildDemented",
                                             "MildDemented",
                                             "ModerateDemented"]},
    ),
    "HeartFailure": Masks(id=45,
                          radlex_id="RID34795",
                          source_names={"PadChest": ["heart insufficiency"]},
                          ),
}
