"""Unit tests for AddProvenance: additive license / source-attribution JSONL fields."""

import os
import tempfile

import jsonlines

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.add_provenance import AddProvenance


def _make_ctx(tmp: str, dataset_name: str = "synthetic") -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name=dataset_name, modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_jsonl(tmp: str, dataset_name: str) -> str:
    """Write a single-record JSONL for the given dataset name."""
    dataset_dir = os.path.join(tmp, f"99_{dataset_name}")
    os.makedirs(dataset_dir, exist_ok=True)
    jsonl_path = os.path.join(dataset_dir, f"99_{dataset_name}.jsonl")
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write(
            {
                "umie_path": f"99_{dataset_name}/CT/Images/99_0_1_0.png",
                "dataset_name": dataset_name,
                "dataset_uid": "99",
                "modality_name": "CT",
                "comparative": "",
                "study_id": "1",
                "umie_id": "99_0_1_0.png",
                "mask_path": "",
                "labels": [],
                "source_labels": [],
            }
        )
    return jsonl_path


def _read_records(jsonl_path: str) -> list:
    """Read all JSONL records from a path."""
    with jsonlines.open(jsonl_path, mode="r") as reader:
        return list(reader)


def test_known_dataset_populated_from_provenance():
    """For a known dataset_name, license/source_dataset come from config.provenance.PROVENANCE."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp, "kits23")
        AddProvenance(_make_ctx(tmp, "kits23")).transform([])

        record = _read_records(jsonl_path)[0]
        assert record["license"] == "CC-BY-NC-SA-4.0"
        assert "KiTS23" in record["source_dataset"]
        # Existing fields preserved.
        assert record["umie_id"] == "99_0_1_0.png"


def test_unknown_dataset_falls_back_to_unknown():
    """An unknown dataset_name falls back to the DEFAULT_PROVENANCE 'unknown' values."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp, "not_a_real_dataset")
        AddProvenance(_make_ctx(tmp, "not_a_real_dataset")).transform([])

        record = _read_records(jsonl_path)[0]
        assert record["license"] == "unknown"
        assert record["source_dataset"] == "unknown"


def test_noop_when_disabled():
    """When add_provenance is False the step is a no-op and adds no fields."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp, "kits23")
        ctx = _make_ctx(tmp, "kits23")
        ctx.metadata.add_provenance = False
        AddProvenance(ctx).transform([])

        record = _read_records(jsonl_path)[0]
        assert "license" not in record
