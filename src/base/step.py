"""Change img ids to match the format of the rest of the dataset."""

from __future__ import annotations

import os
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from sklearn.base import TransformerMixin

from base.creators.xml_mask import BaseXmlMaskCreator
from base.selectors.img_selector import BaseImageSelector
from base.selectors.mask_selector import BaseMaskSelector
from constants import REPORTS_FOLDER_NAME, OutputMode

if TYPE_CHECKING:
    from base.pipeline import (
        ExportConfig,
        FormatConfig,
        MetadataConfig,
        PipelineContext,
        PreprocessingConfig,
        QualityConfig,
    )


class BaseStep(TransformerMixin):
    """Change img ids to match the format of the rest of the dataset."""

    def __init__(self, ctx: PipelineContext):
        """Initialize a Step from the shared pipeline context.

        All formerly-flat arguments are resolved from the named sub-configs on ``ctx``
        (``ctx.paths``, ``ctx.dataset``, ``ctx.identity``, ``ctx.dicom``,
        ``ctx.file_selection``, ``ctx.output``) through the read-only properties below.
        This replaces the previous ~25-parameter signature fed from an ``asdict`` flatten,
        so extractors/selectors are passed as live objects with no serialization round-trip.

        Args:
            ctx (PipelineContext): Structured pipeline configuration shared by all steps.
        """
        self.ctx = ctx
        self.paths_data = np.array([])
        self.json_path = os.path.join(
            self.ctx.paths.target_path,
            f"{self.ctx.dataset.dataset_uid}_{self.ctx.dataset.dataset_name}",
            f"{self.ctx.dataset.dataset_uid}_{self.ctx.dataset.dataset_name}.jsonl",
        )

    # --- Accessors bound to the named sub-configs (the single source of truth) ---
    @property
    def source_path(self) -> str:
        """Path to the source dataset being processed."""
        return self.ctx.paths.source_path

    @property
    def target_path(self) -> str:
        """Path to the directory where the processed dataset is saved."""
        return self.ctx.paths.target_path

    @property
    def labels_path(self) -> Optional[str]:
        """Path to the labels file, if required."""
        return self.ctx.paths.labels_path

    @property
    def masks_path(self) -> Optional[str]:
        """Path to the source masks, if required."""
        return self.ctx.paths.masks_path

    @property
    def output_mode(self) -> OutputMode:
        """Output format for this run: 2D PNG slices (default) or 3D NIfTI volumes."""
        return self.ctx.paths.output_mode

    @property
    def dataset_uid(self) -> str:
        """Unique identifier of the dataset in UMIE."""
        return self.ctx.dataset.dataset_uid

    @property
    def dataset_name(self) -> str:
        """Name of the dataset in UMIE."""
        return self.ctx.dataset.dataset_name

    @property
    def phases(self) -> dict[str, str]:
        """Mapping of phase_id to phase_name."""
        return self.ctx.dataset.phases

    @property
    def labels(self) -> dict[str, list[dict[str, float]]]:
        """Source-label to RadLex-label translation table."""
        return self.ctx.dataset.labels

    @property
    def masks(self) -> dict[str, dict[str, int]]:
        """Mask recolor table as a ``{name: {source_color, target_color}}`` dict view.

        Exposed as plain dicts (rather than the underlying ``MaskColor`` objects) so the
        existing mask-handling code and ``caller.masks[...]`` accesses keep working
        byte-identically with the previous ``asdict`` output.
        """
        return {
            name: {"source_color": mc.source_color, "target_color": mc.target_color}
            for name, mc in self.ctx.dataset.masks.items()
        }

    @property
    def image_folder_name(self) -> str:
        """Name of the folder where images are stored."""
        return self.ctx.output.image_folder_name

    @property
    def mask_folder_name(self) -> str:
        """Name of the folder where masks are stored."""
        return self.ctx.output.mask_folder_name

    @property
    def img_id_extractor(self) -> Callable:
        """Callable extracting the image id from a source path."""
        return self.ctx.identity.img_id_extractor

    @property
    def study_id_extractor(self) -> Callable:
        """Callable extracting the study id from a source path."""
        return self.ctx.identity.study_id_extractor

    @property
    def phase_id_extractor(self) -> Callable:
        """Callable extracting the phase id from a source path."""
        return self.ctx.identity.phase_id_extractor

    @property
    def label_extractor(self) -> Optional[Callable]:
        """Callable producing per-image labels, if any."""
        return self.ctx.identity.label_extractor

    @property
    def zfill(self) -> Optional[int]:
        """Number of digits to pad the image id with."""
        return self.ctx.identity.zfill

    @property
    def window_center(self) -> Optional[int]:
        """DICOM window center used to process images."""
        return self.ctx.dicom.window_center

    @property
    def window_width(self) -> Optional[int]:
        """DICOM window width used to process images."""
        return self.ctx.dicom.window_width

    @property
    def dicom_mapping_attribute(self) -> Optional[str]:
        """DICOM attribute used to map source paths."""
        return self.ctx.dicom.dicom_mapping_attribute

    @property
    def img_prefix(self) -> Optional[str]:
        """Prefix identifying source image file names."""
        return self.ctx.file_selection.img_prefix

    @property
    def segmentation_prefix(self) -> Optional[str]:
        """Prefix identifying source mask file names."""
        return self.ctx.file_selection.segmentation_prefix

    @property
    def mask_prefix(self) -> Optional[str]:
        """String included only in mask file names."""
        return self.ctx.file_selection.mask_prefix

    @property
    def img_selector(self) -> Optional[BaseImageSelector]:
        """Selector identifying intended images."""
        return self.ctx.file_selection.img_selector

    @property
    def mask_selector(self) -> Optional[BaseMaskSelector]:
        """Selector identifying intended masks."""
        return self.ctx.file_selection.mask_selector

    @property
    def multiple_masks_selector(self) -> Optional[dict]:
        """Per-mask selector mapping when there are several masks."""
        return self.ctx.file_selection.multiple_masks_selector

    @property
    def xml_mask_creator(self) -> Optional[BaseXmlMaskCreator]:
        """Creator that builds masks from XML annotations."""
        return self.ctx.file_selection.xml_mask_creator

    # --- Opt-in Theme D-I sub-configs (additive; every step inherits read-only views) ---
    @property
    def quality(self) -> "QualityConfig":
        """Settings for the optional data-quality / validation steps (Theme D)."""
        return self.ctx.quality

    @property
    def preprocessing(self) -> "PreprocessingConfig":
        """Settings for the optional image-preprocessing steps (Theme E)."""
        return self.ctx.preprocessing

    @property
    def metadata_config(self) -> "MetadataConfig":
        """Settings for the optional metadata-enrichment steps (Theme F)."""
        return self.ctx.metadata

    @property
    def format_config(self) -> "FormatConfig":
        """Settings for the optional format / mask conversion steps (Theme G)."""
        return self.ctx.format_conversion

    @property
    def export_config(self) -> "ExportConfig":
        """Settings for the optional reproducibility / export steps (Themes H & I)."""
        return self.ctx.export

    @property
    def dataset_root(self) -> str:
        """Absolute path to this dataset's output root (``{target}/{uid}_{name}``)."""
        return os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")

    def reports_dir(self) -> str:
        """Return (creating if needed) the per-dataset folder for optional analysis reports.

        Used by the opt-in analysis steps (duplicate / corrupt-image / mask-quality / DICOM
        validation / distribution). The folder lives outside ``Images``/``Masks`` so it never
        affects the UMIE id, folder layout, or the golden file-tree of existing pipelines.
        """
        path = os.path.join(self.dataset_root, REPORTS_FOLDER_NAME)
        os.makedirs(path, exist_ok=True)
        return path

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            X (list): List of image paths.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")

    def get_umie_id(self, img_path: str) -> str:
        """Create a unique identifier for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            str: Unique identifier for the image.
        """
        img_id = self.img_id_extractor(img_path)
        if self.output_mode == OutputMode.VOLUMES_3D and img_path.endswith(".nii.gz"):
            # 3D mode: preserve the volumetric extension instead of forcing .png.
            for ext in (".nii.gz", ".gz", ".nii", ".png"):
                if img_id.endswith(ext):
                    img_id = img_id[: -len(ext)]
                    break
            img_id = f"{img_id}.nii.gz"
        else:
            ext = os.path.splitext(img_path)[1]
            img_id = img_id.replace(ext, ".png")

        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_id_extractor(img_path)

        umie_id = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
        return umie_id

    def validate_umie_path(self, img_path: str) -> bool:
        """Check if umie_path is a valid path (a mechanism for discarding imgs).

        Args:
            img_path (str): Path to the image.

        Returns:
            bool: If true, the path is validate.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_id_extractor(img_path)

        if img_id == "" or study_id == "" or phase_id == "":
            return False
        return True

    def get_umie_img_path(self, img_path: str) -> str:
        """Create a unique path for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            str: Unique path for the image.
        """
        umie_id = self.get_umie_id(img_path)

        phase_id = self.phase_id_extractor(img_path)

        if phase_id not in self.phases.keys():
            raise ValueError(f"Phase id {phase_id} not in the list of phases.")
        phase_name = self.phases[phase_id]

        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.image_folder_name,
            umie_id,
        )
        return new_path

    def get_umie_mask_path(self, mask_path: str) -> str:
        """Create a unique path for the mask.

        Args:
            mask_path (str): Path to the mask.

        Returns:
            str: Unique path for the mask.
        """
        umie_id = self.get_umie_id(mask_path)
        phase_id = self.phase_id_extractor(mask_path)

        if phase_id not in self.phases.keys():
            raise ValueError(f"Phase id {phase_id} not in the list of phases.")
        phase_name = self.phases[phase_id]

        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.mask_folder_name,
            umie_id,
        )
        return new_path

    def get_umie_mask_path_from_img_path(self, img_path: str) -> str:
        """Create a unique path for the mask based on image path.

        Args:
            img_path (str): Path to the image.
        Returns:
            str: Unique path for the mask.
        """
        umie_id = os.path.basename(img_path)
        phase_name = self.decode_umie_img_path(img_path)[0]

        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.mask_folder_name,
            umie_id,
        )
        return new_path

    def decode_umie_img_path(self, umie_path: str) -> tuple:
        """Decode the unique image path.

        Args:
            umie_path (str): Unique image path.

        Returns:
            tuple: Tuple containing the phase name, study id, and image id.
        """
        umie_id = os.path.basename(umie_path)
        umie_path_elements = umie_id.split("_")
        phase_name = self.phases[umie_path_elements[1]]
        study_id = umie_path_elements[2]
        img_id = umie_path_elements[3]
        return phase_name, study_id, img_id

    def get_path_without_target_path(self, path: str) -> str:
        """Get the path without the target path.

        Args:
            path (str): Path to the image.

        Returns:
            str: Path to the image without the target path.
        """
        return PureWindowsPath(os.path.relpath(path, self.target_path)).as_posix()
