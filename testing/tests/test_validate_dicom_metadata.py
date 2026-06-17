"""Unit tests for ValidateDicomMetadata using tiny synthetic DICOMs (no external data)."""

import json
import os
import tempfile

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.validate_dicom_metadata import ValidateDicomMetadata


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_dicom(path: str, include_body_part: bool = True) -> None:
    """Write a tiny DICOM, optionally omitting the BodyPartExamined tag."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PixelSpacing = [1.0, 1.0]
    if include_body_part:
        ds.BodyPartExamined = "CHEST"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)


def test_missing_tag_reported_and_not_dropped_by_default():
    """A file missing a required tag is reported, but X is unchanged when not excluding."""
    with tempfile.TemporaryDirectory() as tmp:
        valid = os.path.join(tmp, "valid.dcm")
        missing = os.path.join(tmp, "missing.dcm")
        _write_dicom(valid, include_body_part=True)
        _write_dicom(missing, include_body_part=False)

        ctx = _make_ctx(tmp)
        ctx.quality.required_dicom_tags = {"Modality": "CT", "BodyPartExamined": None}

        returned = ValidateDicomMetadata(ctx).transform([valid, missing])
        assert set(returned) == {valid, missing}  # nothing dropped by default

        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "dicom_metadata_report.json")))
        invalid_keys = set(report["invalid"].keys())
        assert "missing.dcm" in invalid_keys
        assert "valid.dcm" not in invalid_keys


def test_missing_tag_dropped_when_excluding():
    """With exclude_invalid_dicom set, the invalid file is dropped from the returned list."""
    with tempfile.TemporaryDirectory() as tmp:
        valid = os.path.join(tmp, "valid.dcm")
        missing = os.path.join(tmp, "missing.dcm")
        _write_dicom(valid, include_body_part=True)
        _write_dicom(missing, include_body_part=False)

        ctx = _make_ctx(tmp)
        ctx.quality.required_dicom_tags = {"Modality": "CT", "BodyPartExamined": None}
        ctx.quality.exclude_invalid_dicom = True

        returned = ValidateDicomMetadata(ctx).transform([valid, missing])
        assert returned == [valid]


def test_unexpected_value_reported():
    """A required tag with the wrong value is reported as an unexpected value."""
    with tempfile.TemporaryDirectory() as tmp:
        valid = os.path.join(tmp, "valid.dcm")
        _write_dicom(valid, include_body_part=True)

        ctx = _make_ctx(tmp)
        ctx.quality.required_dicom_tags = {"Modality": "MR"}  # actual is CT

        ValidateDicomMetadata(ctx).transform([valid])
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "dicom_metadata_report.json")))
        assert "valid.dcm" in report["invalid"]
        issues = report["invalid"]["valid.dcm"]
        assert any(issue["issue"] == "unexpected_value" for issue in issues)


def test_noop_when_not_configured():
    """When no required tags are configured the step is a no-op and returns X unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        valid = os.path.join(tmp, "valid.dcm")
        _write_dicom(valid, include_body_part=True)
        returned = ValidateDicomMetadata(_make_ctx(tmp)).transform([valid])
        assert returned == [valid]
        assert not os.path.exists(os.path.join(tmp, "99_synthetic", "reports", "dicom_metadata_report.json"))
