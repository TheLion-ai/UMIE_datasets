"""Constants used in the project."""

from enum import Enum

IMG_FOLDER_NAME = "Images"  # name of the folder with images in each target dataset
MASK_FOLDER_NAME = "Masks"  # name of the folder with masks in each target dataset

TARGET_PATH = "./data/"  # name of the folder, where processed outputs will be placed

DATASETS_DOWNLOAD_PATH = "./"


class OutputMode(str, Enum):
    """Per-dataset output format.

    ``SLICES_2D`` keeps the current behavior of writing one PNG per slice. ``VOLUMES_3D``
    (opt-in) preserves volumetric data as NIfTI (``.nii.gz``). See ``UMIE 2D and 3D.md``.

    Subclasses ``str`` so members compare equal by value. This matters because this repo is
    importable under two roots (``constants`` and ``src.constants``) which are loaded as
    distinct modules; a plain ``Enum`` would have non-identical members across the two, but
    a str-based enum compares equal by its string value regardless of which copy is used.
    """

    SLICES_2D = "slices_2d"
    VOLUMES_3D = "volumes_3d"


# Default keeps every existing dataset on the 2D PNG-per-slice workflow (fully backwards compatible).
DEFAULT_OUTPUT_MODE = OutputMode.SLICES_2D
