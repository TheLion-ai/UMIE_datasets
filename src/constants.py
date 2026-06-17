"""Constants used in the project."""

from enum import Enum

IMG_FOLDER_NAME = "Images"  # name of the folder with images in each target dataset
MASK_FOLDER_NAME = "Masks"  # name of the folder with masks in each target dataset
REPORTS_FOLDER_NAME = "reports"  # folder under each dataset's target dir for optional analysis reports

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
    # Combined mode (Theme M, Task 41): write BOTH the 2D PNG slices and the volumetric .nii.gz
    # for the same study, cross-referenced in the JSONL. Default stays SLICES_2D.
    SLICES_2D_AND_VOLUMES_3D = "slices_2d_and_volumes_3d"


# Default keeps every existing dataset on the 2D PNG-per-slice workflow (fully backwards compatible).
DEFAULT_OUTPUT_MODE = OutputMode.SLICES_2D


# JSONL metadata schema versions (Theme K, Task 33). v1 is the flat record currently emitted by
# ``AddUmieIds``; v2 is the richer hierarchical record described in ``JSONlines.md``. The schema is
# selected per-run via ``PathArgs.schema_version`` and defaults to v1, so existing output is
# byte-identical unless a pipeline explicitly opts in to v2.
SCHEMA_VERSION_V1 = "1.0"
SCHEMA_VERSION_V2 = "2.0"
DEFAULT_SCHEMA_VERSION = SCHEMA_VERSION_V1
