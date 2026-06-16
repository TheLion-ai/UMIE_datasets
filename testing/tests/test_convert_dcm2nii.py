"""Unit tests for ConvertDcm2Nii using small synthetic DICOM series (no external data)."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from constants import OutputMode
from src.steps.convert_dcm2nii import ConvertDcm2Nii


def _make_ctx(tmp: str, output_mode: OutputMode = OutputMode.VOLUMES_3D) -> PipelineContext:
    """Build a minimal PipelineContext (no windowing) for the step."""
    identity, dicom, file_selection, output = PipelineArgs().to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp, output_mode=output_mode),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_slice(path, series_uid, instance_number, z, pixels, row_spacing=2.0, col_spacing=1.0):
    """Write one axial CT DICOM slice with the given geometry and int16 pixel array."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.InstanceNumber = instance_number
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # axial
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.PixelSpacing = [row_spacing, col_spacing]  # [between rows (Y), between cols (X)]
    ds.SliceThickness = 5.0
    pixels = pixels.astype(np.int16)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed -> int16
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = pixels.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)


def test_raises_when_not_3d_mode():
    """The step must refuse to run unless output_mode is VOLUMES_3D."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            ConvertDcm2Nii(_make_ctx(tmp, output_mode=OutputMode.SLICES_2D)).transform(["a.dcm"])


def test_series_grouped_ordered_and_affine_correct():
    """Slices are grouped by series, ordered by position, and the affine matches geometry."""
    with tempfile.TemporaryDirectory() as tmp:
        series_uid = generate_uid()
        rows, cols = 3, 4  # Rows=3 (Y), Columns=4 (X)
        # pixel value encodes slice z so we can verify ordering: pix[row,col] = z*1000 + 10*row + col
        zs = [0, 5, 10]
        # write slices in SHUFFLED order to exercise sorting
        for inst, z in [(2, 5), (3, 10), (1, 0)]:
            base = np.fromfunction(lambda r, c: z * 1000 + 10 * r + c, (rows, cols), dtype=int)
            _write_slice(os.path.join(tmp, f"slice_{z}.dcm"), series_uid, inst, z, base)

        out_paths = ConvertDcm2Nii(_make_ctx(tmp)).transform([])
        assert len(out_paths) == 1
        vol = nib.load(out_paths[0])
        data = vol.get_fdata()
        # volume shape is (columns, rows, slices)
        assert data.shape == (cols, rows, len(zs))
        # ordering: slice k corresponds to sorted z; voxel (i=col, j=row, k) == z*1000 + 10*row + col
        for k, z in enumerate(sorted(zs)):
            for j in range(rows):
                for i in range(cols):
                    assert data[i, j, k] == z * 1000 + 10 * j + i, (i, j, k)
        # affine from geometry: col spacing 1 (X), row spacing 2 (Y), z spacing 5
        expected = np.array(
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        np.testing.assert_allclose(vol.affine, expected)
        np.testing.assert_allclose(vol.header.get_zooms()[:3], (1.0, 2.0, 5.0))


def test_two_series_produce_two_volumes():
    """Two distinct SeriesInstanceUIDs are reconstructed into two separate volumes."""
    with tempfile.TemporaryDirectory() as tmp:
        for s in range(2):
            uid = generate_uid()
            for inst, z in [(1, 0), (2, 4)]:
                pix = np.full((2, 2), s * 100 + z, dtype=np.int16)
                _write_slice(os.path.join(tmp, f"s{s}_z{z}.dcm"), uid, inst, z, pix)
        out_paths = ConvertDcm2Nii(_make_ctx(tmp)).transform([])
        assert len(out_paths) == 2
        for p in out_paths:
            assert nib.load(p).shape == (2, 2, 2)


def test_windowing_applied_when_configured():
    """When window params are set, the reconstructed volume is windowed to 0-255 uint8."""
    with tempfile.TemporaryDirectory() as tmp:
        uid = generate_uid()
        for inst, z in [(1, 0), (2, 5)]:
            pix = np.array([[-1000, 40], [240, 1000]], dtype=np.int16)
            _write_slice(os.path.join(tmp, f"z{z}.dcm"), uid, inst, z, pix)
        ctx = _make_ctx(tmp)
        ctx.dicom.window_center = 40
        ctx.dicom.window_width = 400
        out_paths = ConvertDcm2Nii(ctx).transform([])
        vol = nib.load(out_paths[0])
        assert vol.get_data_dtype() == np.uint8
        data = vol.get_fdata()
        assert data.min() >= 0 and data.max() <= 255
