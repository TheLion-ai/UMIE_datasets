from dataclasses import dataclass, field

from src.pipelines.base_pipeline import BasePipeline


@dataclass
class BrainTumorClassificationPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Classification dataset."""

    name: str = field(default="Brain_Tumor_Detection")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
        ]
    )

    def get_label(self) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
