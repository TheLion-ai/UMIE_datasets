"""This file contains the masks used in the datasets. The masks are defined as mask dataclasses."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Mask:
    """This class represents a mask."""

    id: int
    color: int
    radlex_id: str
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # How the mask is named in other datasets, key - dataset name, value - mask name



Background = Mask(id=0, color=0, radlex_id="")
Kidney = Mask(id=1, color=1, radlex_id="RID205", source_names={"KITS23": ["kidney"],
                                                               "CT-ORG": ["Kidney"]})
Neoplasm = Mask(id=2, color=2, radlex_id="RID3957", source_names={"KITS23": ["kidney tumor"],
                                                                  "Brain_Tumor_Progression": ["Brain tumor"]})
RenalCyst = Mask(id=3, color=3, radlex_id="RID35811", source_names={"KITS23": ["cyst"]})
ViralInfection = Mask(id=4, color=4, radlex_id="RID4687", source_names={"MosMedData": ["CT-1"]})
Lung = Mask(id=5, color=5, radlex_id="RID1301", source_names={"CT-ORG": ["lungs"],
                                                              "Lung_segmentation_from_Chest_X-Rays": ["Lungs"], "Finding_and_Measuring_Lungs_in_CT_Data": ["lungs"]})
BoneOrgan = Mask(id=6, color=6, radlex_id="RID13197", source_names={"CT_ORG": ["Bones"]})
Liver = Mask(id=7, color=7, radlex_id="RID58", source_names={"CT_ORG": ["liver"]})
UrinaryBladder = Mask(id=8, color=8, radlex_id="RID237", source_names={"CT_ORG": ["bladder"]})
Brain = Mask(id=9, color=9, radlex_id="RID6434", source_names={"CT-ORG": ["Brain"]})
Nodule = Mask(id=10, color=10, radlex_id="RID3875", source_names={"LIDC-IDRI": ["Nodule>=3mm", "Nodule<3mm"]})
Lesion = Mask(id=11, color=11, radlex_id="RID38780", source_names={"LIDC-IDRI": ["Non-nodule>=3mm"]}) #That is not a nodule
CalciumScore = Mask(id=12, color=12, radlex_id="RID28808", source_names={"Stanford_COCA": ["coronary_artery_calcium"]})
Metastasis = Mask(id=13, color=13, radlex_id="RID5231", source_names={"StanfordBrainMET": ["brain_metastasis"]})
Hemorrhage = Mask(id=14, color=14, radlex_id="RID4700", source_names={"Brain_with_hemorrhage": ["brain_hemorrhage"]})

# """How to denote each mask in UMIE."""

# mask_encodings = {
#     "background": 0,
#     "malignancy1": 3,
#     "malignancy2": 6,
#     "malignancy3": 9,
#     "malignancy4": 12,
#     "malignancy5": 15,
#     "coronary_artery_calcium": 20,
#     "brain": 50,
#     "brain_metastasis": 60,
#     "hemorrhage": 70,
#     "parenchyma_0_25": 100,
#     "kidney_tumor": 101,
#     "kidney_cyst": 102,
#     "kidney": 127,
#     "bone": 140,
#     "liver": 150,
#     "bladder": 160,
#     "lungs": 170,
#     "normal_appearing_cerebral_cortex": 180,
#     "arteliar_input_function": 190,
#     "normal_appearing white matter": 200,
#     "brain_tumor": 255,
# }
