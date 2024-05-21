
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
    "kidney": Mask(id=1, color=1, radlex_id="RID205", source_names={"KITS23": "kidney"}),
    "neoplasm": Mask(id=2, color=2, radlex_id=""),
    "kidney_cyst": Mask(id=3, color=3, radlex_id=""),
    "metastasis": Mask(id=4, color=4, radlex_id="RID5231"),
    "coronary_artery_calcium": Mask(id=4, color=4, radlex_id=""),  # TODO: change to vascular calcification???
    "nodule": Mask(id=5, color=5, radlex_id="RID3875"),
    "lungs": Mask(id=6, color=170, radlex_id="RID203"),
} 
