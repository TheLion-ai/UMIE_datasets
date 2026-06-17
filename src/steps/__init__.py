"""Initialization for the steps module.

This module imports and makes available all the specific processing steps used in various data processing pipelines.
Each step is a class that performs a specific operation, such as converting image formats, copying masks, or adding identifiers.
These steps are used to construct the pipelines defined in the `pipelines` module.

The optional, opt-in steps for Themes D-I (data quality, preprocessing, metadata enrichment, extra
format support, reproducibility infrastructure, and HuggingFace export) are additive: they default to
off / no-op and never change UMIE ids, the folder layout, or existing JSONL fields.
"""

from .add_labels import AddLabels
from .add_provenance import AddProvenance
from .add_umie_ids import AddUmieIds
from .apply_clahe import ApplyClahe
from .apply_windowing import ApplyWindowing
from .autocrop_borders import AutocropBorders
from .check_mask_quality import CheckMaskQuality
from .combine_multiple_masks import CombineMultipleMasks
from .convert_bbox_to_mask import ConvertBboxToMask
from .convert_dcm2nii import ConvertDcm2Nii
from .convert_dcm2png import ConvertDcm2Png
from .convert_dicom_seg import ConvertDicomSeg
from .convert_jpg2png import ConvertJpg2Png
from .convert_jsonl_to_v2 import ConvertJsonlToV2
from .convert_nii2nii import ConvertNii2Nii
from .convert_nii2png import ConvertNii2Png
from .convert_tif2png import ConvertTif2Png
from .copy_masks import CopyMasks
from .create_blank_masks import CreateBlankMasks
from .create_file_to_dcm_attribute_mapping import CreateFileToDcmAttributeMapping
from .create_file_tree import CreateFileTree
from .create_manifest import CreateManifest
from .create_masks_from_xml import CreateMasksFromXml
from .create_splits import CreateSplits
from .delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from .delete_old_preprocessed_data import DeleteOldPreprocessedData
from .delete_temp_files import DeleteTempFiles
from .delete_temp_png import DeleteTempPng
from .detect_corrupt_images import DetectCorruptImages
from .detect_duplicates import DetectDuplicates
from .export_huggingface import ExportHuggingFace
from .extract_dicom_metadata import ExtractDicomMetadata
from .get_file_paths import GetFilePaths
from .masks_to_binary_colors import MasksToBinaryColors
from .merge_masks import MergeMasks
from .normalize_spacing import NormalizeSpacing
from .recolor_masks import RecolorMasks
from .resize_images import ResizeImages
from .skip_processed import SkipProcessed
from .standardize_bit_depth import StandardizeBitDepth
from .store_source_paths import StoreSourcePaths
from .store_volumes_alongside import StoreVolumesAlongside
from .validate_data import ValidateData
from .validate_dicom_metadata import ValidateDicomMetadata
