"""Preprocessing pipeline for the LIDC-IDRI dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs, XmlKeys
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.create_masks_from_xml import CreateMasksFromXML
from src.steps.delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from src.steps.get_file_paths import GetFilePaths


@dataclass
class LidcIdriPipeline(BasePipeline):
    """Preprocessing pipeline for the LIDC-IDRI dataset."""

    name: str = field(default="LidcIdri")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("create_masks_from_xml", CreateMasksFromXML),
            ("convert_dcm2png", ConvertDcm2Png),
            ("add_new_ids", AddNewIds),
            ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        ]
    )
    dataset_args: DatasetArgs = DatasetArgs(
        window_center=-550,
        window_width=2000,
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(x)))).split("-")[
            -1
        ],
        use_siuid_as_index=True,
        zfill=3,
    )
    xml_keys: XmlKeys = XmlKeys(
        study="LidcReadMessage",
        annotator="readingSession",
        obj="unblindedReadNodule",  # in some dataset we have annotations specifying objects and rois in images corresponding to this nodules
        slice="",  # in this dataset we have only one ROI per image
        slice_id="imageSOP_UID",
        inclusion="inclusion",
        roi="roi",
        point="edgeMap",
        x="xCoord",
        y="yCoord",
    )

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args), xml_keys=self.xml_keys)
