"""Preprocessing pipeline for the Stanford Brain MET dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.extractors import BasePhaseIdExtractor, BaseStudyIdExtractor
from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, brain_met_share
from steps import AddUmieIds, CopyMasks, CreateFileTree, GetFilePaths, RecolorMasks


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
        return os.path.basename(os.path.dirname(img_path))


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
    )
    dataset_args: DatasetArgs = brain_met_share
    pipeline_args: PipelineArgs = PipelineArgs(
        zfill=3,
        study_id_extractor=StudyIdExtractor(),
        # Phase name is the folder one level above the image
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
        self.args["phase_id_extractor"] = PhaseIdExtractor(self.args["phases"])
