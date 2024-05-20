from dataclasses import dataclass, field, asdict
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.create_file_tree import CreateFileTree


@dataclass
class BrainTumorClassificationPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Classification dataset."""

    name: str = field(default="Brain_Tumor_Classification_MRI")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
        ]
    )

    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            image_folder_name="Images",
            mask_folder_name=None,
        )
    )

    def get_label(self) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
