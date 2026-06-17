"""
Base pipeline class.

Each source dataset in UMIE has its own pipeline.
Each pipeline is based on the BasePipeline class.
The pipeline is constructed from a set of steps, where each step is a class that processes the data in a specific way.
The pipeline is constructed from the steps and arguments passed to the pipeline.
The pipeline is used to process the dataset and save the processed dataset to the target directory.
"""

import warnings
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, ClassVar, Optional

from sklearn.pipeline import Pipeline

from base.creators import BaseXmlMaskCreator
from base.extractors import (
    BaseImgIdExtractor,
    BaseModalityIdExtractor,
    BaseStudyIdExtractor,
)
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from base.step import BaseStep
from config.dataset_config import DatasetArgs
from src.constants import (
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SCHEMA_VERSION,
    IMG_FOLDER_NAME,
    MASK_FOLDER_NAME,
    SCHEMA_VERSION_V2,
    OutputMode,
)


@dataclass
class PathArgs:
    """Path arguments. These arguments are user defined in runner_config."""

    source_path: str  # path to the source dataset that is being processed, needs to be filled in by the user
    target_path: str  # path to the directory where the processed dataset will be saved, usually TARGET_PATH
    labels_path: Optional[str] = None  # path to the labels file if it is required
    masks_path: Optional[str] = None  # path to the source masks file if it is required
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE  # 2D PNG slices (default) or 3D NIfTI volumes
    schema_version: str = DEFAULT_SCHEMA_VERSION  # JSONL schema: "1.0" flat (default) or "2.0" hierarchical


@dataclass
class IdentityConfig:
    """How to extract UMIE ids (and per-image labels) from source file paths."""

    img_id_extractor: BaseImgIdExtractor = BaseImgIdExtractor()  # function to extract image id from the image path
    study_id_extractor: BaseStudyIdExtractor = BaseStudyIdExtractor()  # function to extract study id
    modality_id_extractor: BaseModalityIdExtractor = BaseModalityIdExtractor({})  # function to extract modality
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
class QualityConfig:
    """Optional settings for the data-quality / validation steps (Theme D).

    Every field has a safe default so the steps are opt-in and behave predictably when a
    pipeline does not configure them. None of these steps alter UMIE ids or folder layout.
    """

    # Task 10 - duplicate / near-duplicate detection
    duplicate_hash_size: int = 8  # side length of the perceptual-hash grid (hash is hash_size**2 bits)
    duplicate_threshold: int = 5  # max Hamming distance between hashes to treat two images as near-duplicates
    flag_duplicates_in_jsonl: bool = False  # additionally write a duplicate_group_id field into the JSONL
    duplicate_reference_hashes: Optional[str] = None  # path to a JSON of {umie_path: hash} for cross-dataset overlap

    # Task 11 - corrupt image detection
    blank_std_threshold: float = 1.0  # pixel-std below which a frame is considered blank (all-black/all-white)
    expected_min_size: Optional[tuple] = None  # (height, width) minimum; smaller images are reported as suspicious

    # Task 13 - DICOM metadata validation
    required_dicom_tags: Optional[dict] = None  # {tag_name: expected_value_or_None}; checked before conversion
    exclude_invalid_dicom: bool = False  # drop files failing the DICOM-tag check from the returned file list


@dataclass
class PreprocessingConfig:
    """Optional, opt-in image preprocessing settings (Theme E).

    All transforms are off by default; they only touch pixel data, never naming/layout, and
    never masks unless explicitly noted. Masks are always resampled/resized nearest-neighbour.
    """

    # Task 14 - intensity windowing / normalization
    window_preset: Optional[str] = None  # named CT preset: lung | bone | soft_tissue | brain | abdomen | mediastinum

    # Task 15 - CLAHE / histogram equalization
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)

    # Task 16 - resolution / pixel-spacing normalization
    target_spacing_mm: Optional[tuple] = None  # e.g. (1.0, 1.0, 1.0) for 1mm isotropic

    # Task 17 - resize with aspect-ratio preservation
    target_size: Optional[tuple] = None  # (height, width)
    resize_strategy: str = "letterbox"  # pad | crop | letterbox | stretch

    # Task 18 - bit-depth standardization
    target_bit_depth: Optional[int] = None  # 8 or 16

    # Task 19 - background / border auto-crop
    autocrop_enabled: bool = False
    autocrop_tolerance: int = 0  # pixels up to this value are treated as background when detecting borders

    # Task 40 - orientation-preserving 2D slicing: write a per-volume geometry sidecar (affine,
    # sform/qform, voxel sizes, slicing axis) next to the slices so they map back to 3D space.
    preserve_slice_geometry: bool = False


@dataclass
class MetadataConfig:
    """Optional metadata-enrichment settings (Theme F). All JSONL additions are additive."""

    # Task 20 - structured DICOM metadata extraction
    dicom_tags: Optional[list] = None  # DICOM tag names to extract into the JSONL (PHI tags are skipped)
    deidentify: bool = True  # never write known-PHI tags; shift dates by study_date_offset_days
    metadata_sidecar: bool = False  # write a sidecar json instead of enriching the JSONL records

    # Task 21 - patient-level reproducible splits
    split_ratios: tuple = (0.7, 0.15, 0.15)  # train / val / test
    split_seed: int = 42
    split_manifest_only: bool = False  # write a split manifest only, do not add a `split` field to the JSONL
    stratify_by_label: bool = True  # keep label distribution similar across splits where possible

    # Task 23 - license / provenance tracking
    add_provenance: bool = True  # populate license/source_dataset fields from config/provenance.py


@dataclass
class FormatConfig:
    """Optional settings for the additional format / mask conversion steps (Theme G)."""

    # Task 24 - DICOM-SEG / RTSTRUCT extraction
    segmentation_structure_map: Optional[dict] = None  # {source_structure_name: target_mask_color}

    # Task 25 - bbox (COCO / YOLO / VOC) -> mask conversion
    bbox_format: Optional[str] = None  # coco | yolo | voc
    bbox_class_map: Optional[dict] = None  # {source_class_id_or_name: target_mask_color}
    bbox_as_filled: bool = True  # filled boxes (True) vs. outline only (False)

    # Task 26 - multi-class mask merging
    overlap_policy: str = "report"  # report | first | last | priority
    merge_priority: Optional[list] = None  # mask colors in descending priority for overlap_policy="priority"


@dataclass
class ExportConfig:
    """Optional reproducibility / export settings (Themes H & I)."""

    # Task 27 - checksums + manifest
    write_manifest: bool = True  # write a sha256 manifest of all outputs
    verify_manifest: bool = False  # re-check outputs against an existing manifest instead of writing one

    # Task 28 - incremental processing
    force_reprocess: bool = False  # ignore existing outputs and reprocess everything

    # Task 29 - parallel execution
    num_workers: int = 1  # >1 enables multiprocessing in steps that support it (deterministic regardless of count)

    # Task 30 - HuggingFace export
    hf_export_path: Optional[str] = None  # local directory to write the Arrow dataset + card to


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
    modality_id_extractor: BaseModalityIdExtractor = BaseModalityIdExtractor(
        {}
    )  # function to extract modality from the image path
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
    # Optional sub-configs for the additive, opt-in steps (Themes D-I). Passed as whole objects
    # (rather than flat fields) to keep PipelineArgs from ballooning; default to None -> defaults.
    quality: Optional[QualityConfig] = None
    preprocessing: Optional[PreprocessingConfig] = None
    metadata: Optional[MetadataConfig] = None
    format_conversion: Optional[FormatConfig] = None
    export: Optional[ExportConfig] = None

    def to_configs(self) -> tuple[IdentityConfig, DicomConfig, FileSelectionConfig, OutputConfig]:
        """Group the flat pipeline args into the four focused sub-configs (no value change).

        This lets pipelines keep building a single flat ``PipelineArgs`` (backwards
        compatible) while ``PipelineContext`` exposes the purpose-driven views.
        """
        return (
            IdentityConfig(
                img_id_extractor=self.img_id_extractor,
                study_id_extractor=self.study_id_extractor,
                modality_id_extractor=self.modality_id_extractor,
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
    # Optional, opt-in sub-configs for Themes D-I. Defaulted so every existing direct
    # construction of PipelineContext (e.g. in unit tests) keeps working unchanged.
    quality: QualityConfig = field(default_factory=QualityConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    format_conversion: FormatConfig = field(default_factory=FormatConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


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

    # 2D conversion step name -> its 3D equivalent step name. The 3D classes are resolved
    # lazily (see _resolve_3d_step) to avoid an import cycle with the steps package.
    STEP_3D_ALTERNATIVES: ClassVar[dict[str, str]] = {
        "convert_nii2png": "convert_nii2nii",
        "convert_dcm2png": "convert_dcm2nii",
    }

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
            # Opt-in Theme D-I sub-configs: use what the pipeline supplied, else safe defaults.
            quality=self.pipeline_args.quality or QualityConfig(),
            preprocessing=self.pipeline_args.preprocessing or PreprocessingConfig(),
            metadata=self.pipeline_args.metadata or MetadataConfig(),
            format_conversion=self.pipeline_args.format_conversion or FormatConfig(),
            export=self.pipeline_args.export or ExportConfig(),
        )

        # Inherently-2D datasets (X-ray, pre-sliced) have no volumetric conversion step and cannot
        # honor a 3D or combined request: warn and fall back to 2D so the run neither crashes nor
        # produces empty output. Combined mode (Task 41) also needs volumetric source data.
        volumetric_modes = (OutputMode.VOLUMES_3D, OutputMode.SLICES_2D_AND_VOLUMES_3D)
        if self.ctx.paths.output_mode in volumetric_modes and not self._supports_3d():
            warnings.warn(
                f"Dataset '{self.name}' has no volumetric conversion step; "
                f"output_mode={self.ctx.paths.output_mode.value} is ignored and falls back to 2D slices.",
                stacklevel=2,
            )
            self.ctx.paths.output_mode = OutputMode.SLICES_2D

        self.args: dict = dict(**asdict(self.path_args), **asdict(self.dataset_args))

        # Run prepare_pipeline only if source path exists.
        if self.args["source_path"]:
            self.prepare_pipeline()

    def _supports_3d(self) -> bool:
        """Return True if any step has a registered 3D alternative (i.e. dataset is volumetric)."""
        return any(step_name in self.STEP_3D_ALTERNATIVES for step_name, _ in self.steps)

    @staticmethod
    def _resolve_3d_step(step_name: str) -> type:
        """Resolve a 3D step name to its class (lazy import avoids a base<->steps import cycle)."""
        from steps import ConvertDcm2Nii, ConvertNii2Nii

        registry = {"convert_nii2nii": ConvertNii2Nii, "convert_dcm2nii": ConvertDcm2Nii}
        return registry[step_name]

    @property
    def pipeline(self) -> Pipeline:
        """Create a pipeline, swapping 2D conversion steps for 3D ones in VOLUMES_3D mode.

        Each step receives the structured PipelineContext (no flat-dict / asdict). When
        ``output_mode == VOLUMES_3D``, ``convert_nii2png``/``convert_dcm2png`` are replaced by
        ``convert_nii2nii``/``convert_dcm2nii`` per ``STEP_3D_ALTERNATIVES``. When
        ``schema_version == "2.0"``, a final ``convert_jsonl_to_v2`` step is appended so the emitted
        JSONL is upgraded to the hierarchical v2 schema (no-op / not appended in the v1 default).
        """
        use_3d = self.ctx.paths.output_mode == OutputMode.VOLUMES_3D
        steps = []
        for step_name, step_cls in self.steps:
            if use_3d and step_name in self.STEP_3D_ALTERNATIVES:
                step_name_3d = self.STEP_3D_ALTERNATIVES[step_name]
                steps.append((step_name_3d, self._resolve_3d_step(step_name_3d)(self.ctx)))
            else:
                steps.append((step_name, step_cls(self.ctx)))
        # Combined mode (Task 41): keep the 2D PNG conversion above and additionally preserve the
        # volumetric .nii.gz alongside the slices.
        if self.ctx.paths.output_mode == OutputMode.SLICES_2D_AND_VOLUMES_3D:
            steps.append(("store_volumes_alongside", self._resolve_step("StoreVolumesAlongside")(self.ctx)))
        if self.ctx.paths.schema_version == SCHEMA_VERSION_V2:
            steps.append(("convert_jsonl_to_v2", self._resolve_step("ConvertJsonlToV2")(self.ctx)))
        return Pipeline(steps=steps)

    @staticmethod
    def _resolve_step(class_name: str) -> type:
        """Resolve an optional step class by name lazily (avoids a base<->steps import cycle)."""
        import steps as steps_module

        return getattr(steps_module, class_name)

    @abstractmethod
    def prepare_pipeline(self) -> None:
        """Prepare pipeline. Function is called in post initialization if source path exists."""
        return
