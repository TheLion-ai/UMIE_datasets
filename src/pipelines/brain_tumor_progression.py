"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import (
    BaseImgIdExtractor,
    BasePhaseIdExtractor,
    BaseStudyIdExtractor,
)
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, brain_tumor_progression
from steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CopyMasks,
    CreateBlankMasks,
    CreateFileTree,
    DeleteTempFiles,
    GetFilePaths,
    RecolorMasks,
    StoreSourcePaths,
    ValidateData,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return self._extract_by_separator(img_path, "-")


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Getting study id depends on location of the file.
        # Study_id is retrieved in a different way when image already is moved to target directory with new name.
        return self._extract_parent_dir(img_path, parent_dir_level=-2, include_path=False)[-5:]


class PhaseIdExtractor(BasePhaseIdExtractor):
    """Extractor for phase IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str, *args: Any) -> str:
        """Extract phase id from img path."""
        # dot in folder name breaks the code
        img_path = img_path.replace(".", "-")
        phase_name = self._extract_parent_dir(img_path=img_path, parent_dir_level=1, include_path=False).split("-")[-2]

        return self._get_phase_id_from_dict(phase_name)


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Brain Tumor Progression dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return "MaskTumor" not in path


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Brain Tumor Progression dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return "MaskTumor" in path


@dataclass
class BrainTumorProgressionPipeline(BasePipeline):
    """Preprocessing pipeline for the Brain Tumor Progression dataset."""

    name: str = field(default="brain_tumor_progression")  # dataset name used in configs
    steps: tuple = (
        ("get_file_paths", GetFilePaths),
        ("create_file_tree", CreateFileTree),
        ("store_source_paths", StoreSourcePaths),
        ("convert_dcm2png", ConvertDcm2Png),
        ("copy_png_masks", CopyMasks),
        ("recolor_masks", RecolorMasks),
        ("add_umie_ids", AddUmieIds),
        # optionally delete images with empty masks
        # ("delete_imgs_with_no_annotations", DeleteImgsWithNoAnnotations),
        # or create blank masks
        ("create_blank_masks", CreateBlankMasks),
        ("delete_temp_files", DeleteTempFiles),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_progression)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            # Image id is in the source file name after the last underscore
            img_id_extractor=ImgIdExtractor(),  # lambda x: os.path.basename(x).split("-")[-1],
            # Study name is the folder two levels above the image
            study_id_extractor=StudyIdExtractor(),
            mask_prefix="MaskTumor",
            segmentation_prefix="MaskTumor",
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        # self.pipeline_args.img_id_extractor = lambda x: img_id_extractor(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["phase_id_extractor"] = PhaseIdExtractor(self.args["phases"])
