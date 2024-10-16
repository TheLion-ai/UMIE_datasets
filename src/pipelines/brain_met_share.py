"""Preprocessing pipeline for the Stanford Brain MET dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BasePhaseIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from config.dataset_config import DatasetArgs, brain_met_share
from steps import (
    AddUmieIds,
    CopyMasks,
    CreateFileTree,
    GetFilePaths,
    RecolorMasks,
    ValidateData,
)


class StudyIdExtractor(BaseStudyIdExtractor):
    """Extractor for study IDs specific to the Brain MET dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract study id from img path."""
        # Study name is the folder two levels above the image
        study_id = os.path.basename(os.path.dirname(os.path.dirname(img_path))).split("_")[-1]
        return study_id


class PhaseIdExtractor(BasePhaseIdExtractor):
    """Extractor for phase IDs specific to the Brain MET dataset."""

    def _extract(self, img_path: str) -> str:
        """Extract phase id from img path."""
        # Phase name is the folder one level above the image
        phase_name = os.path.basename(os.path.dirname(img_path))
        phase_id = [key for key, value in self.phases.items() if value == phase_name][0]
        return str(phase_id)


class ImageSelector(BaseImageSelector):
    """Selector for images specific to the Brain MET dataset."""

    def _is_image_file(self, path: str) -> bool:
        """Check if the file is the intended image."""
        return True


class MaskSelector(BaseMaskSelector):
    """Selector for masks specific to the Brain MET dataset."""

    def _is_mask_file(self, path: str) -> bool:
        """Check if the file is the intended mask."""
        return True


@dataclass
class BrainMETSharePipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford Brain MET dataset."""

    name: str = "brain_met_share"  # dataset name used in configs
    steps: tuple = (
        ("create_file_tree", CreateFileTree),
        ("get_file_paths", GetFilePaths),
        ("copy_png_masks", CopyMasks),
        ("add_new_ids", AddUmieIds),
        ("recolor_masks", RecolorMasks),
        ("validate_data", ValidateData),
    )
    dataset_args: DatasetArgs = field(
        default_factory=lambda: brain_met_share
    )  # Update default value to use default_factory
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=3,
            study_id_extractor=StudyIdExtractor(),
            img_selector=ImageSelector(),
            mask_selector=MaskSelector(),
            # Phase name is the folder one level above the image
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["phase_id_extractor"] = PhaseIdExtractor(self.args["phases"])
