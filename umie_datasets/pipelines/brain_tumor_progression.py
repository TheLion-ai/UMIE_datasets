"""Preprocessing pipeline for the Stanford COCA dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from umie_datasets.base.extractors import (
    BaseImgIdExtractor,
    BasePhaseIdExtractor,
    BaseStudyIdExtractor,
)
from umie_datasets.base.pipeline import BasePipeline, PipelineArgs
from umie_datasets.config.dataset_config import DatasetArgs, brain_tumor_progression
from umie_datasets.steps import (
    AddUmieIds,
    ConvertDcm2Png,
    CopyMasks,
    CreateBlankMasks,
    CreateFileTree,
    DeleteTempFiles,
    GetFilePaths,
    RecolorMasks,
    StoreSourcePaths,
)


class ImgIdExtractor(BaseImgIdExtractor):
    """Extractor for image IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return os.path.basename(img_path).split("-")[-1]


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Getting study id depends on location of the file.
        # Study_id is retrieved in a different way when image already is moved to target directory with new name.
        return os.path.basename(os.path.dirname(os.path.dirname(img_path)))[-5:]


class PhaseIdExtractor(BasePhaseIdExtractor):
    """Extractor for phase IDs specific to the Brain Tumor Progression dataset."""

    def _extract(self, img_path: str, *args: Any) -> str:
        """Extract phase id from img path."""
        for phase in self.phases.keys():
            if self.phases[phase] in img_path:
                return str(phase)
        return ""


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
    )
    dataset_args: DatasetArgs = field(default_factory=lambda: brain_tumor_progression)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=4,
            # Image id is in the source file name after the last underscore
            img_id_extractor=ImgIdExtractor(),  # lambda x: os.path.basename(x).split("-")[-1],
            # Study name is the folder two levels above the image
            study_id_extractor=StudyIdExtractor(),
            mask_selector="MaskTumor",
            segmentation_prefix="MaskTumor",
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        # self.pipeline_args.img_id_extractor = lambda x: img_id_extractor(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["phase_id_extractor"] = PhaseIdExtractor(self.args["phases"])
