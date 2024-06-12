"""Preprocessing pipeline for the Stanford Brain MET dataset."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from config import dataset_config
from src.base.pipeline import BasePipeline, PipelineArgs
from steps.add_umie_ids import AddUmieIds
from src.steps.copy_masks import CopyMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths
from src.steps.recolor_masks import RecolorMasks


@dataclass
class BrainMETSharePipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford Brain MET dataset."""

    name: str = field(default="brain_met_share")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("copy_png_masks", CopyMasks),
            ("add_new_ids", AddUmieIds),
            ("recolor_masks", RecolorMasks),
        ]
    )
    dataset_args: dataset_config.brain_met_share = field(default_factory=lambda: dataset_config.brain_met_share)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            zfill=3,
            # Study name is the folder two levels above the image
            study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))).split("_")[-1],
            # Phase name is the folder one level above the image
            phase_extractor=lambda x: os.path.basename(os.path.dirname(x)),
        )
    )

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
