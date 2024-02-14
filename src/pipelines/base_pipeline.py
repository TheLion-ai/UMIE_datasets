"""Base pipeline class."""
import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from config import (
    dataset_masks_config,
    dataset_uids_config,
    mask_encodings_config,
    phases_config,
)
from src.constants import IMG_FOLDER_NAME, MASK_FOLDER_NAME


@dataclass
class PathArgs:
    """Path arguments. These arguments are user defined."""

    source_path: str  # path to the dataset that is being processed
    target_path: str  # path to the directory where the processed dataset will be saved
    labels_path: Optional[str]  # path to the labels file
    masks_path: Optional[str]  # path to the source masks file


@dataclass
class DatasetArgs:
    """Dataset arguments. These arguments are dataset specific."""

    image_folder_name: Optional[str] = IMG_FOLDER_NAME  # name of folder, where images will be stored
    mask_folder_name: Optional[str | None] = MASK_FOLDER_NAME  # name of folder, where masks will be stored
    zfill: Optional[int] = None  # number of digits to pad the image id with
    img_id_extractor: Optional[Callable] = lambda x: os.path.basename(
        x
    )  # function to extract image id from the image path
    study_id_extractor: Optional[Callable] = lambda x: x  # function to extract study id from the image path
    phase_extractor: Optional[Callable] = lambda x: "0"  # function to extract phase from the image path
    window_center: Optional[int] = None  # value used to process DICOM images
    window_width: Optional[int] = None  # value used to process DICOM images
    get_label: Optional[Callable] = None  # function to get label for the individual image
    img_dcm_prefix: Optional[str] = None  # prefix of the source image file names
    segmentation_dcm_prefix: Optional[str] = "segmentation"  # prefix of the source mask file names


@dataclass  # type: ignore[misc]
class BasePipeline:
    """The base class for constructing a pipeline for an individual dataset."""

    path_args: PathArgs  # arguments passed to the pipeline, user defines them at each run of the pipeline
    name: str  # name of the dataset
    dataset_args: DatasetArgs  # arguments passed to sklearn pipeline required to process a specific dataset
    steps: list[tuple[str, TransformerMixin]]
    args: dict = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Prepare args for the pipeline."""
        self.load_config()

        # Run prepare_pipeline only if source path exists.
        if self.args["source_path"]:
            self.prepare_pipeline()

    @property
    def pipeline(self) -> None:
        """Create a pipeline based on the steps and args."""
        args = self.args
        # Create pipeline and pass args to each step
        return Pipeline(steps=[(step[0], step[1](**args)) for step in self.steps])

    def load_config(self) -> None:
        """Load configuration files from config."""
        # Unique identifier of the dataset
        dataset_uid = dataset_uids_config.dataset_uids[self.name]
        # Dict with phases of the dataset
        phases = phases_config.phases[self.name]
        # Dict with masks and the source colors encoding
        dataset_masks = dataset_masks_config.dataset_masks[self.name]
        # Dict with source colors of masks and the target colors mapping
        mask_colors_source2target = {v: mask_encodings_config.mask_encodings[k] for k, v in dataset_masks.items()}
        # Dict with args extracted from the dataset config
        cfg_args = {
            "dataset_name": self.name,
            "dataset_uid": dataset_uid,
            "phases": phases,
            "dataset_masks": dataset_masks,
            "mask_colors_source2target": mask_colors_source2target,
        }
        # Update args with the config args
        self.args = dict(**self.path_args, **cfg_args)

    def load_labels_from_path(self) -> list:
        """Load all labels from the labels file. Labels are not processed here, they are processed in the get_label method.

        Returns:
            list: List of labels and annotations.
        """
        labels_path_extension = os.path.basename(self.args["labels_path"]).split(".")[1]
        if labels_path_extension == "json":
            with open(self.args["labels_path"]) as f:
                labels_list = json.load(f)
                return labels_list
        else:
            raise NotImplementedError(f"Labels path extention {labels_path_extension} is not supported.")

    @abstractmethod
    def prepare_pipeline(self) -> None:
        """Prepare pipeline. Function is called in __post_init__ if source path exists."""
        return

    @abstractmethod
    def get_label(self) -> list:
        """Get label for the image. Method is implemented in the dataset specific pipeline."""
        return []
