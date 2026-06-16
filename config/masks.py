"""
This file contains the masks used in the datasets. The masks are defined as mask dataclasses.

Each mask has the following attributes:
- id: int - the id of the mask
- radlex_name: str - the name of the mask in RadLex
- color: int - the color of the mask
- radlex_id: str - the id of the mask in RadLex
- source_names: dict - the names of the mask in other datasets, meant to be helpful when translating new source dataset masks to UMIE.
"""

from dataclasses import dataclass, field
from typing import Optional

from config import ontology
from config.ontology import OntologyTerm


@dataclass
class Mask:
    """This class represents a mask."""

    id: int  # The unique id of the mask in UMIE (also the value of the mask in the mask image)
    radlex_name: str  # The name of the mask in RadLex
    color: int  # The color of the mask in the mask image
    radlex_id: str  # The id of the mask in RadLex
    source_names: Optional[dict] = field(
        default_factory=dict
    )  # How the mask is named in other datasets, key - dataset name, value - source mask name in this dataset
    # --- Theme L additions (all optional; never affect ``id``/``color``/``radlex_name``) ---
    radlex_parent_id: Optional[str] = None  # RadLex id of the parent concept (Task 36 hierarchy)
    anatomy: Optional[OntologyTerm] = None  # multi-axis decomposition (Task 37): the body part
    pathology: Optional[OntologyTerm] = None  # multi-axis decomposition (Task 37): the abnormality
    modifier: Optional[OntologyTerm] = None  # multi-axis decomposition (Task 37): grade / qualifier
    snomed_id: Optional[str] = None  # secondary ontology cross-code (Task 39): SNOMED CT
    fma_id: Optional[str] = None  # secondary ontology cross-code (Task 39): Foundational Model of Anatomy
    uberon_id: Optional[str] = None  # secondary ontology cross-code (Task 39): Uberon


Background = Mask(
    id=0, radlex_name="Background", color=0, radlex_id=""
)  # Actually there is no radlex for backgroud ofc but "Normality descriptor" didnt fit either

Kidney = Mask(
    id=1,
    radlex_name="Kidney",
    color=1,
    radlex_id="RID205",
    anatomy=OntologyTerm(radlex_name="Kidney", radlex_id="RID205"),  # Task 37: organ mask is its own anatomy
    snomed_id="64033007",  # SNOMED CT: Kidney structure (Task 39)
    fma_id="FMA:7203",  # Foundational Model of Anatomy: Kidney
    uberon_id="UBERON:0002113",  # Uberon: kidney
    source_names={"kits23": ["kidney"], "ct_org": ["Kidney"]},
)

Neoplasm = Mask(
    id=2,
    radlex_name="Neoplasm",
    color=2,
    radlex_id="RID3957",
    radlex_parent_id=None,  # root of the neoplasm subtree (Task 36)
    pathology=OntologyTerm(radlex_name="Neoplasm", radlex_id="RID3957"),  # Task 37
    snomed_id="108369006",  # SNOMED CT: Neoplasm and/or hamartoma (Task 39)
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
    anatomy=OntologyTerm(radlex_name="Lung", radlex_id="RID1301"),  # Task 37
    snomed_id="39607008",  # SNOMED CT: Lung structure (Task 39)
    fma_id="FMA:7195",
    uberon_id="UBERON:0002048",
    source_names={
        "ct_org": ["lungs"],
        "chest_xray_masks_and_label": ["Lungs"],
        "finding_and_measuring_lungs": ["lungs"],
    },
)

BoneOrgan = Mask(id=6, radlex_name="BoneOrgan", color=6, radlex_id="RID13197", source_names={"ct_org": ["Bones"]})

Liver = Mask(
    id=7,
    radlex_name="Liver",
    color=7,
    radlex_id="RID58",
    anatomy=OntologyTerm(radlex_name="Liver", radlex_id="RID58"),  # Task 37
    snomed_id="10200004",  # SNOMED CT: Liver structure (Task 39)
    fma_id="FMA:7197",
    uberon_id="UBERON:0002107",
    source_names={"ct_org": ["liver"]},
)

UrinaryBladder = Mask(
    id=8, radlex_name="UrinaryBladder", color=8, radlex_id="RID237", source_names={"ct_org": ["bladder"]}
)

Brain = Mask(
    id=9,
    radlex_name="Brain",
    color=9,
    radlex_id="RID6434",
    anatomy=OntologyTerm(radlex_name="Brain", radlex_id="RID6434"),  # Task 37
    snomed_id="12738006",  # SNOMED CT: Brain structure (Task 39)
    fma_id="FMA:50801",
    uberon_id="UBERON:0000955",
    source_names={"ct_org": ["Brain"]},
)

Nodule = Mask(
    id=10,
    radlex_name="Nodule",
    color=10,
    radlex_id="RID3875",
    source_names={"lidc_idri": ["Nodule>=3mm", "Nodule<3mm"]},
)

Lesion = Mask(
    id=11,
    radlex_name="Lesion",
    color=11,
    radlex_id="RID38780",
    source_names={"lidc_idri": ["Non-nodule>=3mm"]},
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
    radlex_parent_id="RID3957",  # parent: Neoplasm (Task 36)
    source_names={"brain_met_share": ["brain_metastasis"]},
)

Hemorrhage = Mask(
    id=14,
    radlex_name="Hemorrhage",
    color=14,
    radlex_id="RID4700",
    source_names={"brain_with_intracranial_hemorrhage": ["brain_hemorrhage"]},
)

Calcification = Mask(
    id=15,
    radlex_name="Calcification",
    color=15,
    radlex_id="RID5196",
    source_names={"Chest_X-ray_Abnormalities_Detection": ["Calcification"], "cmmd": ["calcification"]},
)

Mass = Mask(
    id=16,
    radlex_name="Mass",
    color=16,
    radlex_id="RID3874",
    source_names={"ChestX-ray14": ["Mass"], "cmmd": ["mass"]},
)

# Registry of every defined mask, mirroring ``config/labels.py``'s ``all_labels``. Consumed by the
# cross-dataset distribution report (Task 22) and the mask-quality check (Task 12); also fixes the
# previously-missing ``config.masks.all_masks`` reference in ``utils/data_counter.py``.
all_masks = [obj for name, obj in list(globals().items()) if isinstance(obj, Mask)]


# --- Theme L ontology query helpers (Task 36), scoped to the mask registry --------------------
def mask_by_name(name: str) -> Optional[Mask]:
    """Return the mask with the given RadLex name, or ``None``."""
    return ontology.get_by_name(name, all_masks)


def mask_ancestors(mask: Mask) -> list[Mask]:
    """Return ``mask``'s ancestors, nearest parent first."""
    return ontology.ancestors(mask, all_masks)


def mask_descendants(mask: Mask) -> list[Mask]:
    """Return every transitive descendant of ``mask``."""
    return ontology.descendants(mask, all_masks)


def organ_masks() -> list[Mask]:
    """Return the masks that denote an anatomical structure (their ``anatomy`` axis is set)."""
    return [m for m in all_masks if m.anatomy is not None and not m.anatomy.is_empty()]


def validate_masks() -> list[str]:
    """Audit the mask registry (duplicate / empty / dangling RadLex ids, cycles). Empty == clean."""
    return ontology.validate(all_masks)
