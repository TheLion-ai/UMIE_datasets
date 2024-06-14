"""Preprocessing pipeline for the Stanford Brain MET dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from base.pipeline import BasePipeline, PipelineArgs
from config.dataset_config import DatasetArgs, brain_met_share
from steps import AddUmieIds, CopyMasks, CreateFileTree, GetFilePaths, RecolorMasks


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
        # Study name is the folder two levels above the image
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))).split("_")[-1],
        # Phase name is the folder one level above the image
        phase_extractor=lambda x: os.path.basename(os.path.dirname(x)),
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
