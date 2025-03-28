"""Preprocessing pipeline for CBIS-DDSM dataset."""
import glob
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from base.extractors import (
    BaseImgIdExtractor,
    BaseLabelExtractor,
    BasePhaseIdExtractor,
    BaseStudyIdExtractor,
)
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, cbis_ddsm
from steps import (
    AddLabels,
    AddUmieIds,
    CombineMultipleMasks,
    ConvertDcm2Png,
    CopyMasks,
    CreateFileTree,
    DeleteTempFiles,
    DeleteTempPng,
    GetFilePaths,
    StoreSourcePaths,
)
from steps.combine_multiple_masks_with_group import CombineMultipleMasksWithGroup


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the CBIS-DDSM dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._return_zero()


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the CBIS-DDSM dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))

        return study_id


class PhaseIdExtractor(BasePhaseIdExtractor):
    """Extractor for phase IDs specific to the CBIS-DDSM dataset."""

    def _extract(self, img_path: str, *args: Any) -> str:
        """Return phase id."""
        return "0"


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the CBIS-DDSM dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return not bool(re.search(r"_[1-7]", path))


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the CBIS-DDSM dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return bool(re.search(r"_[1-7]", path)) and os.path.basename(path)[-7:][:3] == "1-2"


class LabelExtractor(BaseLabelExtractor):
    """Extractor for labels specific to the CBIS-DDSM dataset."""

    def __init__(self, labels: dict[str, list[int]], labels_path: os.PathLike):
        """Extract labels from the source image path."""
        super().__init__(labels)
        csv_files = glob.glob(os.path.join(labels_path, "**/*.csv"), recursive=True)
        self.calc_test_set = pd.read_csv(csv_files[0])
        self.calc_train_set = pd.read_csv(csv_files[1])
        self.mass_test_set = pd.read_csv(csv_files[2])
        self.mass_train_set = pd.read_csv(csv_files[3])

    def _extract(self, img_path: str, mask_path: str) -> list:
        """Extract label from img path."""
        study_id = StudyIdExtractor()._extract(img_path)
        img_dataset = study_id.split("_")[0]
        print(img_dataset)
        if "Calc-Test" == img_dataset:
            labels_df = self.calc_test_set
        elif "Calc-Train" == img_dataset:
            labels_df = self.calc_train_set
        elif "Mass-Test" == img_dataset:
            labels_df = self.mass_test_set
        elif "Mass-Train" == img_dataset:
            labels_df = self.mass_train_set
        else:
            raise Exception("No labels found for the image")

        radlex_labels: list[dict] = []

        parts = study_id.split("_")
        print(parts)
        patient_id = "P_" + parts[2]
        left_or_right = parts[3]
        image_view = parts[4]

        case = labels_df[
            (labels_df["patient_id"] == patient_id)
            & (labels_df["left or right breast"] == left_or_right)
            & (labels_df["image view"] == image_view)
        ]

        print(case["pathology"].values)
        pathology = case["pathology"].values[0]
        bi_rads = case["assessment"].values[0]

        print(str(bi_rads))

        radlex_labels.append(self.labels[pathology])
        radlex_labels.append(self.labels[str(bi_rads)])

        return radlex_labels


@dataclass
class CbisDdsmPipeline(BasePipeline):
    """Preprocessing pipeline for the Cbis_ddsm dataset."""

    name: str = field(default="cbis_ddsm")  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("store_source_paths", StoreSourcePaths),
        ("convert_dcm2png", ConvertDcm2Png),
        ("combine_masks", CombineMultipleMasksWithGroup),
        ("copy_masks", CopyMasks),
        ("add_umie_ids", AddUmieIds),
        ("add_labels", AddLabels),
        ("delete_temp_files", DeleteTempFiles),
        ("delete_temp_png", DeleteTempPng),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: cbis_ddsm)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            img_id_extractor=ImgIdExtractor(),
            study_id_extractor=StudyIdExtractor(),
            mask_selector=MaskSelector(),
            img_selector=ImageSelector(),
            window_center=24083,  # 12041 ,#41452
            window_width=53493,
            segmentation_prefix="_",
            mask_prefix="_",
            multiple_masks_selector={
                "_1": "Neoplasm",
                "_2": "Neoplasm",
                "_3": "Neoplasm",
                "_4": "Neoplasm",
                "_5": "Neoplasm",
                "_6": "Neoplasm",
                "_7": "Neoplasm",
            },
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Update args with dataset_args
        self.pipeline_args.label_extractor = LabelExtractor(self.args["labels"], self.args["labels_path"])
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        # self.args["label_extractor"] = LabelExtractor(self.args["labels"], self.args["labels_path"])
