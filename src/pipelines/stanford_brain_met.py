"""Preprocessing pipeline for the Stanford Brain MET dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_new_ids import AddNewIds
from src.steps.copy_png_masks import CopyPNGMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths
from src.steps.recolor_masks import RecolorMasks


@dataclass
class StanfordBrainMETPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford Brain MET dataset."""

    name: str = field(default="StanfordBrainMET")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("copy_png_masks", CopyPNGMasks),
            ("add_new_ids", AddNewIds),
            ("recolor_masks", RecolorMasks),
        ]
    )
    dataset_args: DatasetArgs = DatasetArgs(
        zfill=3,
        # Study name is the folder two levels above the image
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))).split("_")[-1],
        # Phase name is the folder one level above the image
        phase_extractor=lambda x: os.path.basename(os.path.dirname(x)),
    )

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
