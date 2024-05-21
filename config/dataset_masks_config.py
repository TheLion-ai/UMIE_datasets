"""This file contains the masks used in the datasets. The masks are defined as mask dataclasses."""
from dataclasses import dataclass, field
from typing import Optional

# 0	nic ciekawego
# 255	Brain tumor
# 200	NAWM(Normal Appearing White Matter)
# 180	NACC(Normal Appearing Cerebral Cortex)
# 50	mózg
# 190	AIFx3 (Arteliar input function)
# 100	Parenchyma 0-25%
# 3	Malignancy1
# 6	Malignancy2
# 9	Malignancy3
# 12	Malignancy4
# 15	Malignancy5
# 127	Kidneys
# 150	liver
# 160	Bladder
# 170	lungs
# 140	Bone
# 101	Góz nerki
# 60	brain metastasis
mask_encodings = {
    "background": 0,
    "malignancy1": 3,
    "malignancy2": 6,
    "malignancy3": 9,
    "malignancy4": 12,
    "malignancy5": 15,
    "coronary_artery_calcium": 20,
    "brain": 50,
    "brain_metastasis": 60,
    "hemorrhage": 70,
    "parenchyma_0_25": 100,
    "kidney_tumor": 101,
    "kidney_cyst": 102,
    "kidney": 127,
    "bone": 140,
    "liver": 150,
    "bladder": 160,
    "lungs": 170,
    "normal_appearing_cerebral_cortex": 180,
    "arteliar_input_function": 190,
    "normal_appearing white matter": 200,
    "brain_tumor": 255,
}


dataset_masks = {
    "StanfordCOCA": {
        "coronary_artery_calcium": 20,
    },
    "StanfordBrainMET": {
        "brain_metastasis": 255,
    },
    "KITS23": {"kidney": 1, "kidney_tumor": 2, "kidney_cyst": 3},
    "CoronaHack_Chest_X-Ray_Dataset": {},
    "Brain_with_hemorrhage": {
        "hemorrhage": 255,
    },
    "Alzheimers_Dataset": {},
    "Brain_Tumor_Detection": {},
    "Covid19_Detection": {},
    "Finding_and_Measuring_Lungs_in_CT_Data": {
        "lungs": 255,
    },
    "Liver_And_Liver_Tumor": {"liver": 1, "liver_tumor": 1},
    "ChestX-ray14": {},
    "Knee_Osteoarthritis": {},
    "Brain_Tumor_Classification_MRI": {},
}


@dataclass
class Mask:
    """This class represents a mask."""

    id: int
    color: int
    radlex_id: str
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # How the mask is named in other datasets, key - dataset name, value - mask name


masks = {
    "background": Mask(id=0, color=0, radlex_id=""),
    "kidney": Mask(id=1, color=1, radlex_id="RID205"),
    "kidney_tumor": Mask(id=2, color=2, radlex_id=""),
    "kidney_cyst": Mask(id=3, color=3, radlex_id=""),
    "metastasis": Mask(id=4, color=4, radlex_id="RID5231"),
    "coronary_artery_calcium": Mask(id=4, color=4, radlex_id=""),  # TODO: change to vascular calcification???
    "nodule": Mask(id=5, color=5, radlex_id="RID3875"),
    "lungs": Mask(id=6, color=170, radlex_id="RID203"),
}
