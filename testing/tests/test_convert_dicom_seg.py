"""Unit tests for ConvertDicomSeg helpers using tiny synthetic Datasets (no external data)."""

import tempfile

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.convert_dicom_seg import ConvertDicomSeg


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def test_patient_to_pixel_identity_orientation_unit_spacing():
    """Identity orientation and unit spacing map patient (x, y) directly to (col, row)."""
    image_position = [0.0, 0.0, 0.0]
    image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # axis-aligned
    pixel_spacing = [1.0, 1.0]

    assert ConvertDicomSeg._patient_to_pixel((3.0, 4.0, 0.0), image_position, image_orientation, pixel_spacing) == (
        4,
        3,
    )
    assert ConvertDicomSeg._patient_to_pixel((0.0, 0.0, 0.0), image_position, image_orientation, pixel_spacing) == (
        0,
        0,
    )


def test_rasterize_contour_axis_aligned_square():
    """An axis-aligned square contour fills the expected pixel bbox with the given colour."""
    image_position = [0.0, 0.0, 0.0]
    image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pixel_spacing = [1.0, 1.0]
    # square with corners (2,2)..(6,6) in patient coords (x, y, z)
    square = [(2.0, 2.0, 0.0), (6.0, 2.0, 0.0), (6.0, 6.0, 0.0), (2.0, 6.0, 0.0)]

    mask = ConvertDicomSeg._rasterize_contour(
        square, image_position, image_orientation, pixel_spacing, rows=10, cols=10, color=5
    )
    assert mask[4, 4] == 5  # interior filled
    assert mask[2, 2] == 5  # corner of the square
    assert mask[0, 0] == 0  # outside the square
    assert mask[8, 8] == 0
    # filled region spans the expected pixel bbox
    rows, cols = np.where(mask == 5)
    assert rows.min() == 2 and cols.min() == 2
    assert rows.max() == 6 and cols.max() == 6


def _segment(number: int, label: str) -> Dataset:
    """Build a minimal SegmentSequence item."""
    item = Dataset()
    item.SegmentNumber = number
    item.SegmentLabel = label
    return item


def _frame_group(segment_number: int) -> Dataset:
    """Build a minimal PerFrameFunctionalGroups item referencing a segment number."""
    seg_id = Dataset()
    seg_id.ReferencedSegmentNumber = segment_number
    group = Dataset()
    group.SegmentIdentificationSequence = [seg_id]
    return group


def _seg_dataset(frames: np.ndarray) -> Dataset:
    """Build a minimal multi-frame SEG Dataset whose pixel_array decodes (file_meta set)."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.PixelData = frames.tobytes()
    ds.Rows = int(frames.shape[1])
    ds.Columns = int(frames.shape[2])
    ds.NumberOfFrames = int(frames.shape[0])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    return ds


def test_seg_to_masks_two_frame_binary():
    """A 2-frame binary SEG yields a mask per frame with each segment's colour where pixels==1."""
    with tempfile.TemporaryDirectory() as tmp:
        step = ConvertDicomSeg(_make_ctx(tmp))

        # two binary frames, one per segment
        frame0 = np.zeros((4, 4), dtype=np.uint8)
        frame0[0:2, 0:2] = 1
        frame1 = np.zeros((4, 4), dtype=np.uint8)
        frame1[2:4, 2:4] = 1
        ds = _seg_dataset(np.stack([frame0, frame1]))
        # two segments mapped to config/masks RadLex names -> colours Kidney=1, Neoplasm=2
        ds.SegmentSequence = [_segment(1, "Kidney"), _segment(2, "Neoplasm")]
        ds.PerFrameFunctionalGroupsSequence = [_frame_group(1), _frame_group(2)]

        masks = step._seg_to_masks(ds)
        # no DerivationImageSequence -> frames keyed by index
        assert set(masks.keys()) == {0, 1}
        assert masks[0][0, 0] == 1  # segment 1 -> Kidney colour
        assert masks[0][3, 3] == 0
        assert masks[1][3, 3] == 2  # segment 2 -> Neoplasm colour
        assert masks[1][0, 0] == 0


def test_seg_to_masks_uses_structure_map_override():
    """A configured segmentation_structure_map overrides the config/masks fallback colour."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.segmentation_structure_map = {"MyTumor": 13}
        step = ConvertDicomSeg(ctx)

        frame0 = np.zeros((4, 4), dtype=np.uint8)
        frame0[0:2, 0:2] = 1
        ds = _seg_dataset(frame0[np.newaxis, ...])
        ds.SegmentSequence = [_segment(1, "MyTumor")]
        ds.PerFrameFunctionalGroupsSequence = [_frame_group(1)]

        masks = step._seg_to_masks(ds)
        assert masks[0][0, 0] == 13  # mapped via structure map override
