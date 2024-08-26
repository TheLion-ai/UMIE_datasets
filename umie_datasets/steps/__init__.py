"""Initialization for the steps module.

This module imports and makes available all the specific processing steps used in various data processing pipelines.
Each step is a class that performs a specific operation, such as converting image formats, copying masks, or adding identifiers.
These steps are used to construct the pipelines defined in the `pipelines` module.
"""


from .add_labels import AddLabels
from .add_umie_ids import AddUmieIds
from .combine_multiple_masks import CombineMultipleMasks
from .convert_dcm2png import ConvertDcm2Png
from .convert_jpg2png import ConvertJpg2Png
from .convert_nii2png import ConvertNii2Png
from .convert_tif2png import ConvertTif2Png
from .copy_masks import CopyMasks
from .create_blank_masks import CreateBlankMasks
from .create_file_tree import CreateFileTree
from .create_masks_from_xml import CreateMasksFromXML
from .delete_imgs_with_no_annotations import DeleteImgsWithNoAnnotations
from .delete_temp_files import DeleteTempFiles
from .delete_temp_png import DeleteTempPng
from .get_file_paths import GetFilePaths
from .masks_to_binary_colors import MasksToBinaryColors
from .recolor_masks import RecolorMasks
from .store_source_paths import StoreSourcePaths
