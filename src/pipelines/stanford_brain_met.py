"""Preprocessing pipeline for the Stanford Brain MET dataset."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.preprocessing.add_new_ids import AddNewIds
from src.preprocessing.copy_png_masks import CopyPNGMasks
from src.preprocessing.create_file_tree import CreateFileTree
from src.preprocessing.get_file_paths import GetFilePaths
from src.preprocessing.recolor_masks import RecolorMasks


@dataclass
class StanfordBrainMETPipeline(BasePipeline):
    """Preprocessing pipeline for the Stanford Brain MET dataset."""

    name: str = field(default="Stanford_Brain_MET")
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
        study_id_extractor=lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))).split("_")[-1],
        phase_extractor=lambda x: os.path.basename(os.path.dirname(x)),
    )

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
