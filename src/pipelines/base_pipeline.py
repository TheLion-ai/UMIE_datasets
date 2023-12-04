"""Base pipeline class."""
import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import yaml
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


@dataclass
class PathArgs:
    """Path arguments."""

    source_path: str
    target_path: str
    labels_path: Optional[str]


@dataclass
class DatasetArgs:
    """Dataset arguments."""

    zfill: Optional[int] = None
    img_id_extractor: Optional[Callable] = lambda x: os.path.basename(x)
    study_id_extractor: Optional[Callable] = lambda x: x
    phase_extractor: Optional[Callable] = lambda x: x
    window_center: Optional[int] = None
    window_width: Optional[int] = None
    get_label: Optional[Callable] = None
    img_dcm_prefix: Optional[str] = None
    segmentation_dcm_prefix: Optional[str] = None


@dataclass  # type: ignore[misc]
class BasePipeline:
    """Base pipeline class."""

    path_args: PathArgs
    name: str
    dataset_args: DatasetArgs
    steps: list[tuple[str, TransformerMixin]]
    args: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Post initialization actions."""
        self.load_config()

    @property
    def pipeline(self) -> None:
        """Create a pipeline."""
        args = self.args

        return Pipeline(steps=[(step[0], step[1](**args)) for step in self.steps])

    def load_config(self) -> None:
        """Load configuration."""
        dataset_uid = yaml.load(open("config/dataset_uid_config.yaml"), Loader=yaml.FullLoader)[self.name]
        phases = yaml.load(open("config/phases_config.yaml"), Loader=yaml.FullLoader)[self.name]
        mask_encoding_config = yaml.load(open("config/masks_encoding_config.yaml"), Loader=yaml.FullLoader)
        dataset_masks = yaml.load(open("config/dataset_masks_config.yaml"), Loader=yaml.FullLoader)[self.name]
        mask_colors_old2new = {v: mask_encoding_config[k] for k, v in dataset_masks.items()}

        cfg_args = {
            "dataset_name": self.name,
            "dataset_uid": dataset_uid,
            "phases": phases,
            "dataset_masks": dataset_masks,
            "mask_colors_old2new": mask_colors_old2new,
        }
        self.args = dict(**self.path_args, **cfg_args)

    def load_labels_from_path(self) -> list:
        """Load labels from the path.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of labels.
        """
        labels_path_extention = os.path.basename(self.args["labels_path"]).split(".")[1]
        if labels_path_extention == "json":
            with open(self.args["labels_path"]) as f:
                labels_list = json.load(f)
                return labels_list
        else:
            raise NotImplementedError(f"Labels path extention {labels_path_extention} is not supported.")

    @abstractmethod
    def get_label(self) -> list:
        """Get label for the image."""
        return []
