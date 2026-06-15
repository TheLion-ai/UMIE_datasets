"""
Base pipeline class.

Each source dataset in UMIE has its own pipeline.
Each pipeline is based on the BasePipeline class.
The pipeline is constructed from a set of steps, where each step is a class that processes the data in a specific way.
The pipeline is constructed from the steps and arguments passed to the pipeline.
The pipeline is used to process the dataset and save the processed dataset to the target directory.
"""

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from sklearn.pipeline import Pipeline

from base.creators import BaseXmlMaskCreator
from base.extractors import (
    BaseImgIdExtractor,
    BasePhaseIdExtractor,
    BaseStudyIdExtractor,
)
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from base.step import BaseStep
from config.dataset_config import DatasetArgs
from src.constants import DEFAULT_OUTPUT_MODE, IMG_FOLDER_NAME, MASK_FOLDER_NAME, OutputMode


@dataclass
class PathArgs:
    """Path arguments. These arguments are user defined in runner_config."""

    source_path: str  # path to the source dataset that is being processed, needs to be filled in by the user
    target_path: str  # path to the directory where the processed dataset will be saved, usually TARGET_PATH
    labels_path: Optional[str] = None  # path to the labels file if it is required
    masks_path: Optional[str] = None  # path to the source masks file if it is required
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE  # 2D PNG slices (default) or 3D NIfTI volumes


@dataclass
class IdentityConfig:
    """How to extract UMIE ids (and per-image labels) from source file paths."""

    img_id_extractor: BaseImgIdExtractor = BaseImgIdExtractor()  # function to extract image id from the image path
    study_id_extractor: BaseStudyIdExtractor = BaseStudyIdExtractor()  # function to extract study id
    phase_id_extractor: BasePhaseIdExtractor = BasePhaseIdExtractor({})  # function to extract phase
    label_extractor: Optional[Callable] = None  # function to get label for the individual image
    zfill: Optional[int] = None  # number of digits to pad the image id with


@dataclass
class DicomConfig:
    """DICOM-specific windowing / mapping parameters."""

    window_center: Optional[int] = None  # value used to process DICOM images
    window_width: Optional[int] = None  # value used to process DICOM images
    dicom_mapping_attribute: Optional[str] = None  # dicom attribute to map paths to


@dataclass
class FileSelectionConfig:
    """How to identify images vs. masks in the source dataset."""

    img_prefix: Optional[str] = None  # prefix of the source image file names
    segmentation_prefix: Optional[str] = None  # prefix of the source mask file names
    mask_prefix: Optional[str] = None  # string included only in masks names
    img_selector: Optional[BaseImageSelector] = None  # function to select intended images by path
    mask_selector: Optional[BaseMaskSelector] = None  # function to select intended masks by path
    multiple_masks_selector: Optional[dict] = None  # selector + meaning per mask when there are several
    xml_mask_creator: Optional[BaseXmlMaskCreator] = None  # function to create masks from xml files


@dataclass
class OutputConfig:
    """Target directory naming."""

    image_folder_name: str = IMG_FOLDER_NAME  # name of folder, where images will be stored
    mask_folder_name: str = MASK_FOLDER_NAME  # name of folder, where masks will be stored


@dataclass
class PipelineArgs:
    """
    Arguments required by the processing pipeline of a given dataset.

    These arguments are pipeline specific.
    Not all datasets require all arguments. Most arguments are optional or have default values.
    """

    image_folder_name: str = IMG_FOLDER_NAME  # name of folder, where images will be stored
    mask_folder_name: str = MASK_FOLDER_NAME  # name of folder, where masks will be stored
    img_id_extractor: BaseImgIdExtractor = BaseImgIdExtractor()  # function to extract image id from the image path
    study_id_extractor: BaseStudyIdExtractor = (
        BaseStudyIdExtractor()
    )  # function to extract study id from the image path
    phase_id_extractor: BasePhaseIdExtractor = BasePhaseIdExtractor({})  # function to extract phase from the image path
    mask_selector: BaseMaskSelector = None  # function to select intended masks by path
    img_selector: BaseImageSelector = None  # function to select intended images by path
    zfill: Optional[int] = None  # number of digits to pad the **image id** with
    window_center: Optional[int] = None  # value used to process DICOM images
    window_width: Optional[int] = None  # value used to process DICOM images
    label_extractor: Optional[Callable] = None  # function to get label for the individual image
    img_prefix: Optional[str] = (
        None  # prefix of the source image file names, used to differentiate between images and masks
    )
    segmentation_prefix: Optional[str] = (
        None  # prefix of the source mask file names, used to differentiate between images and masks
    )
    mask_prefix: Optional[str] = (
        None  # string included only in masks names, used to differentiate between images and masks
    )
    multiple_masks_selector: Optional[dict] = (
        None  # dict including mask selector and its meaning for each mask, used when there are multiple masks and each mask has a different selector
    )
    xml_mask_creator: Optional[BaseXmlMaskCreator] = None  # function to create masks from xml files
    dicom_mapping_attribute: Optional[str] = None  # dicom attribute to map paths to

    def to_configs(self) -> tuple[IdentityConfig, DicomConfig, FileSelectionConfig, OutputConfig]:
        """Group the flat pipeline args into the four focused sub-configs (no value change).

        This lets pipelines keep building a single flat ``PipelineArgs`` (backwards
        compatible) while ``PipelineContext`` exposes the purpose-driven views.
        """
        return (
            IdentityConfig(
                img_id_extractor=self.img_id_extractor,
                study_id_extractor=self.study_id_extractor,
                phase_id_extractor=self.phase_id_extractor,
                label_extractor=self.label_extractor,
                zfill=self.zfill,
            ),
            DicomConfig(
                window_center=self.window_center,
                window_width=self.window_width,
                dicom_mapping_attribute=self.dicom_mapping_attribute,
            ),
            FileSelectionConfig(
                img_prefix=self.img_prefix,
                segmentation_prefix=self.segmentation_prefix,
                mask_prefix=self.mask_prefix,
                img_selector=self.img_selector,
                mask_selector=self.mask_selector,
                multiple_masks_selector=self.multiple_masks_selector,
                xml_mask_creator=self.xml_mask_creator,
            ),
            OutputConfig(
                image_folder_name=self.image_folder_name,
                mask_folder_name=self.mask_folder_name,
            ),
        )


@dataclass
class PipelineContext:
    """Structured wrapper exposing the pipeline args as purpose-driven sub-configs.

    Part of the pipeline-args reorganization (see ``New pipeline args organization.md``).
    The sub-configs (``identity``, ``dicom``, ``file_selection``, ``output``) are derived
    from the pipeline's flat ``PipelineArgs`` so steps can be moved onto
    ``ctx.<group>.<field>`` access. Introducing it does not change behavior - the flat
    ``args`` dict built in ``BasePipeline`` is still what the steps consume until Task 4.
    """

    paths: PathArgs
    dataset: DatasetArgs
    identity: IdentityConfig
    dicom: DicomConfig
    file_selection: FileSelectionConfig
    output: OutputConfig


@dataclass  # type: ignore[misc]
class BasePipeline:
    """The base class for constructing a pipeline for an individual dataset."""

    path_args: PathArgs  # arguments passed to the pipeline, user defines them at each run of the pipeline
    dataset_args: DatasetArgs
    name: str  # name of the dataset
    pipeline_args: (
        PipelineArgs  # arguments passed to sklearn pipeline required to process a specific dataset # defined in config
    )
    steps: tuple[tuple[str, BaseStep]]

    def __post_init__(self) -> None:
        """Prepare args for the pipeline."""
        # Structured view of the args (no behavior change; the flat dict below is still what
        # the steps consume). The sub-configs are decomposed from pipeline_args.
        identity, dicom, file_selection, output = self.pipeline_args.to_configs()
        self.ctx = PipelineContext(
            paths=self.path_args,
            dataset=self.dataset_args,
            identity=identity,
            dicom=dicom,
            file_selection=file_selection,
            output=output,
        )

        self.args: dict = dict(**asdict(self.path_args), **asdict(self.dataset_args))

        # Run prepare_pipeline only if source path exists.
        if self.args["source_path"]:
            self.prepare_pipeline()

    @property
    def pipeline(self) -> Pipeline:
        """Create a pipeline based on the steps and the shared context."""
        # Each step now receives the structured PipelineContext (no flat-dict / asdict).
        return Pipeline(steps=[(step[0], step[1](self.ctx)) for step in self.steps])

    @abstractmethod
    def prepare_pipeline(self) -> None:
        """Prepare pipeline. Function is called in post initialization if source path exists."""
        return
