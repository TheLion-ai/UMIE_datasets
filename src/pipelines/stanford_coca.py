"""Preprocessing pipeline for the Stanford COCA dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.convert_dcm2png import ConvertDcm2Png
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.create_masks_from_xml import CreateMasksFromXML
from src.preprocessing.delete_imgs_with_no_annotations import DeleteImgsWithoutMasks
from src.preprocessing.get_file_paths import GetFilePaths


@dataclass
class StanfordCOCAPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford COCA dataset."""

    name: str = field(default="Stanford_COCA")
    steps: list = field(
        default_factory=lambda: [
            ("get_file_paths", GetFilePaths),
            ("create_file_tree", CreateFileTree),
            ("add_new_ids", AddNewIds),
            ("convert_dcm2png", ConvertDcm2Png),
            ("create_masks_from_xml", CreateMasksFromXML),
            # Choose either to create blank masks or delete images without masks
            # ("create_blank_masks", CreateBlankMasks),
            ("delete_imgs_without_masks", DeleteImgsWithoutMasks),
        ],
    )
    dataset_args: DatasetArgs = DatasetArgs(
        zfill=4,
        img_id_extractor=lambda x: os.path.basename(x).split("-")[-1],
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
    )

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
