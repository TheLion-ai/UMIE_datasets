"""This file contains the masks used in the datasets. The masks are defined as mask dataclasses."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mask:
    """This class represents a mask."""

    id: int
    radlex_name: str
    color: int
    radlex_id: str
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # How the mask is named in other datasets, key - dataset name, value - mask name


Background = Mask(
    id=0, radlex_name="Background", color=0, radlex_id=""
)  # Actually there is no radlex for backgroud ofc but "Normality descriptor" didnt fit either

Kidney = Mask(
    id=1, radlex_name="Kidney", color=1, radlex_id="RID205", source_names={"kits23": ["kidney"], "ct_org": ["Kidney"]}
)

Neoplasm = Mask(
    id=2,
    radlex_name="Neoplasm",
    color=2,
    radlex_id="RID3957",
    source_names={"kits23": ["kidney tumor"], "brain_tumor_progression": ["Brain tumor"], "lits": ["liver_tumor"]},
)
RenalCyst = Mask(id=3, radlex_name="RenalCyst", color=3, radlex_id="RID35811", source_names={"kits23": ["cyst"]})
ViralInfection = Mask(
    id=4, radlex_name="ViralInfection", color=4, radlex_id="RID4687", source_names={"mos_med_data": ["CT-1"]}
)

Lung = Mask(
    id=5,
    radlex_name="Lung",
    color=5,
    radlex_id="RID1301",
    source_names={
        "ct_org": ["lungs"],
        "chest_xray_masks_and_label": ["Lungs"],
        "finding_and_measuring_lungs": ["lungs"],
    },
)

BoneOrgan = Mask(id=6, radlex_name="BoneOrgan", color=6, radlex_id="RID13197", source_names={"ct_org": ["Bones"]})

Liver = Mask(id=7, radlex_name="Liver", color=7, radlex_id="RID58", source_names={"ct_org": ["liver"]})

UrinaryBladder = Mask(
    id=8, radlex_name="UrinaryBladder", color=8, radlex_id="RID237", source_names={"ct_org": ["bladder"]}
)

Brain = Mask(id=9, radlex_name="Brain", color=9, radlex_id="RID6434", source_names={"ct_org": ["Brain"]})

Nodule = Mask(
    id=10,
    radlex_name="Nodule",
    color=10,
    radlex_id="RID3875",
    source_names={"lidc_idri": ["Nodule>=3mm", "Nodule<3mm"]},
)

Lesion = Mask(
    id=11, radlex_name="Lesion", color=11, radlex_id="RID38780", source_names={"lidc_idri": ["Non-nodule>=3mm"]}
)  # That is not a nodule

CalciumScore = Mask(
    id=12,
    radlex_name="CalciumScore",
    color=12,
    radlex_id="RID28808",
    source_names={"coca": ["coronary_artery_calcium"]},
)

Metastasis = Mask(
    id=13,
    radlex_name="Metastasis",
    color=13,
    radlex_id="RID5231",
    source_names={"brain_met_share": ["brain_metastasis"]},
)

Hemorrhage = Mask(
    id=14,
    radlex_name="Hemorrhage",
    color=14,
    radlex_id="RID4700",
    source_names={"brain_with_intracranial_hemorrhage": ["brain_hemorrhage"]},
)
