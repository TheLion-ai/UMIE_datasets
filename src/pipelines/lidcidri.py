"""Preprocessing pipeline for the LIDC-IDRI dataset."""
from dataclasses import asdict, dataclass, field
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_dcm2png import ConvertDcm2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths


@dataclass
class LidcIdriPipeline(BasePipeline):
    """Preprocessing pipeline for the LIDC-IDRI dataset."""

    name: str = field(default="LidcIdri")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("convert_dcm2png", ConvertDcm2Png),
            ("add_new_ids", AddNewIds),
        ]
    )
    dataset_args: DatasetArgs = DatasetArgs(
        window_center=-550,
        window_width=2000,
    )

    def __post_init__(self) -> None:
        """Post initialization actions."""
        super().__post_init__()
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
