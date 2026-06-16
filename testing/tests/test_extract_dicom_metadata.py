"""Unit tests for ExtractDicomMetadata using tiny synthetic DICOMs (no external data)."""

import json
import os
import tempfile

import jsonlines
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.extract_dicom_metadata import ExtractDicomMetadata


def _make_ctx(tmp: str, dataset_name: str = "synthetic") -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name=dataset_name, phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_dicom(path: str) -> None:
    """Write a tiny DICOM with a mix of technical tags and a PHI tag (PatientName)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientSex = "F"
    ds.SliceThickness = "1.5"
    ds.KVP = "120"
    ds.StudyDate = "20200115"
    ds.PatientName = "DOE^JANE"  # PHI - must never be written when de-identifying
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path)


def _setup_dataset(tmp: str) -> tuple[str, str]:
    """Create a JSONL record, a matching umie image file, a DICOM, and source_paths.json.

    Returns:
        tuple[str, str]: (jsonl_path, umie_abs_image_path).
    """
    dataset_dir = os.path.join(tmp, "99_synthetic")
    images_dir = os.path.join(dataset_dir, "CT", "Images")
    os.makedirs(images_dir, exist_ok=True)

    umie_id = "99_0_1_0.png"
    umie_abs = os.path.join(images_dir, umie_id)
    with open(umie_abs, "wb") as handle:
        handle.write(b"fake-png")
    umie_path = os.path.relpath(umie_abs, tmp)

    dcm_path = os.path.join(tmp, "source_0.dcm")
    _write_dicom(dcm_path)

    with open(os.path.join(tmp, "source_paths.json"), "w") as handle:
        json.dump({umie_abs: dcm_path}, handle)

    jsonl_path = os.path.join(dataset_dir, "99_synthetic.jsonl")
    record = {
        "umie_path": umie_path,
        "dataset_name": "synthetic",
        "dataset_uid": "99",
        "phase_name": "CT",
        "comparative": "",
        "study_id": "1",
        "umie_id": umie_id,
        "mask_path": "",
        "labels": [],
        "source_labels": [],
    }
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write(record)
    return jsonl_path, umie_path


def _read_records(jsonl_path: str) -> list:
    """Read all JSONL records from a path."""
    with jsonlines.open(jsonl_path, mode="r") as reader:
        return list(reader)


def test_requested_non_phi_fields_written_phi_omitted():
    """Requested technical tags appear in the JSONL; the PHI tag (PatientName) never does."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path, umie_path = _setup_dataset(tmp)
        ctx = _make_ctx(tmp)
        ctx.metadata.dicom_tags = ["PatientSex", "SliceThickness", "KVP", "PatientName"]

        ExtractDicomMetadata(ctx).transform([umie_path])

        records = _read_records(jsonl_path)
        assert len(records) == 1
        meta = records[0]["dicom_metadata"]
        assert meta["PatientSex"] == "F"
        assert str(meta["SliceThickness"]) == "1.5"
        assert "KVP" in meta
        assert "PatientName" not in meta  # PHI denylist enforced even though requested


def test_date_tag_is_shifted_not_verbatim():
    """A requested date tag is shifted (not stored verbatim) and records the offset."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path, umie_path = _setup_dataset(tmp)
        ctx = _make_ctx(tmp)
        ctx.metadata.dicom_tags = ["StudyDate"]

        ExtractDicomMetadata(ctx).transform([umie_path])

        meta = _read_records(jsonl_path)[0]["dicom_metadata"]
        assert "StudyDate" in meta
        assert meta["StudyDate"] != "20200115"  # de-identified by a deterministic shift
        assert "StudyDate_offset_days" in meta


def test_sidecar_mode_leaves_jsonl_untouched():
    """With metadata_sidecar, the JSONL is unchanged and a sidecar json is written."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path, umie_path = _setup_dataset(tmp)
        ctx = _make_ctx(tmp)
        ctx.metadata.dicom_tags = ["PatientSex"]
        ctx.metadata.metadata_sidecar = True

        ExtractDicomMetadata(ctx).transform([umie_path])

        assert "dicom_metadata" not in _read_records(jsonl_path)[0]
        sidecar = os.path.join(tmp, "99_synthetic", "reports", "dicom_metadata.json")
        data = json.load(open(sidecar))
        assert data[umie_path]["PatientSex"] == "F"


def test_noop_when_no_tags_configured():
    """When dicom_tags is None the step is a no-op and adds nothing to the JSONL."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path, umie_path = _setup_dataset(tmp)
        ctx = _make_ctx(tmp)  # dicom_tags defaults to None

        returned = ExtractDicomMetadata(ctx).transform([umie_path])
        assert returned == [umie_path]
        assert "dicom_metadata" not in _read_records(jsonl_path)[0]
