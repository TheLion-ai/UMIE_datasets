"""Base pipeline class."""

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from sklearn.pipeline import Pipeline

from base.extractors import (
    BaseImgIdExtractor,
    BasePhaseIdExtractor,
    BaseStudyIdExtractor,
)
from base.step import BaseStep
from config.dataset_config import DatasetArgs
from src.constants import IMG_FOLDER_NAME, MASK_FOLDER_NAME


@dataclass
class PathArgs:
    """Path arguments. These arguments are user defined."""

    source_path: str  # path to the dataset that is being processed
    target_path: str  # path to the directory where the processed dataset will be saved
    labels_path: Optional[str] = None  # path to the labels file
    masks_path: Optional[str] = None  # path to the source masks file


@dataclass
class PipelineArgs:
    """Dataset arguments. These arguments are dataset specific."""

    image_folder_name: str = IMG_FOLDER_NAME  # name of folder, where images will be stored
    mask_folder_name: str = MASK_FOLDER_NAME  # name of folder, where masks will be stored
    img_id_extractor: BaseImgIdExtractor = BaseImgIdExtractor()  # function to extract image id from the image path
    study_id_extractor: BaseStudyIdExtractor = (
        BaseStudyIdExtractor()
    )  # function to extract study id from the image path
    phase_id_extractor: BasePhaseIdExtractor = BasePhaseIdExtractor({})  # function to extract phase from the image path
    zfill: Optional[int] = None  # number of digits to pad the image id with
    window_center: Optional[int] = None  # value used to process DICOM images
    window_width: Optional[int] = None  # value used to process DICOM images
    label_extractor: Optional[Callable] = None  # function to get label for the individual image
    img_prefix: Optional[str] = None  # prefix of the source image file names
    segmentation_prefix: Optional[str] = None  # prefix of the source mask file names
    mask_selector: Optional[str] = None  # string included only in masks names
    multiple_masks_selector: Optional[dict] = None  # dict including mask selector and its meaning for each mask


@dataclass  # type: ignore[misc]
class BasePipeline:
    """The base class for constructing a pipeline for an individual dataset."""

    path_args: PathArgs  # arguments passed to the pipeline, user defines them at each run of the pipeline
    dataset_args: DatasetArgs
    name: str  # name of the dataset
    pipeline_args: PipelineArgs  # arguments passed to sklearn pipeline required to process a specific dataset # defined in config
    steps: tuple[tuple[str, BaseStep]]

    def __post_init__(self) -> None:
        """Prepare args for the pipeline."""
        self.args: dict = dict(**asdict(self.path_args), **asdict(self.dataset_args))

        # Run prepare_pipeline only if source path exists.
        if self.args["source_path"]:
            self.prepare_pipeline()

    @property
    def pipeline(self) -> Pipeline:
        """Create a pipeline based on the steps and args."""
        args = self.args
        # Create pipeline and pass args to each step
        return Pipeline(steps=[(step[0], step[1](**args)) for step in self.steps])

    @abstractmethod
    def prepare_pipeline(self) -> None:
        """Prepare pipeline. Function is called in post initialization if source path exists."""
        return
