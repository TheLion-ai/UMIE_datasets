"""Unit tests for CreateSplits: study-level, reproducible, roughly-ratio'd stratified splits."""

import os
import tempfile

import jsonlines

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.create_splits import CreateSplits


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


def _write_jsonl(tmp: str, num_studies: int = 40, slices_per_study: int = 3) -> str:
    """Write a JSONL with many study ids, several image records each, some labelled."""
    dataset_dir = os.path.join(tmp, "99_synthetic")
    os.makedirs(dataset_dir, exist_ok=True)
    jsonl_path = os.path.join(dataset_dir, "99_synthetic.jsonl")

    with jsonlines.open(jsonl_path, mode="w") as writer:
        for study in range(num_studies):
            # Alternate labels across studies so stratification has something to balance.
            labels = [{"Neoplasm": 1}] if study % 2 == 0 else [{"NormalityDecriptor": 1}]
            for slice_idx in range(slices_per_study):
                writer.write(
                    {
                        "umie_path": f"99_synthetic/CT/Images/99_0_{study}_{slice_idx}.png",
                        "dataset_name": "synthetic",
                        "dataset_uid": "99",
                        "modality_name": "CT",
                        "comparative": "",
                        "study_id": str(study),
                        "umie_id": f"99_0_{study}_{slice_idx}.png",
                        "mask_path": "",
                        "labels": labels,
                        "source_labels": [],
                    }
                )
    return jsonl_path


def _read_records(jsonl_path: str) -> list:
    """Read all JSONL records from a path."""
    with jsonlines.open(jsonl_path, mode="r") as reader:
        return list(reader)


def _study_to_split(records: list) -> dict:
    """Build a {study_id: set_of_splits} map to detect leakage."""
    mapping: dict = {}
    for record in records:
        mapping.setdefault(record["study_id"], set()).add(record["split"])
    return mapping


def test_no_study_leaks_across_splits():
    """Every study_id must appear in exactly one split (critical for 3D volumes)."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp)
        CreateSplits(_make_ctx(tmp)).transform([])

        mapping = _study_to_split(_read_records(jsonl_path))
        for study_id, splits in mapping.items():
            assert len(splits) == 1, f"study {study_id} leaked across splits {splits}"


def test_deterministic_with_same_seed():
    """Running twice with the same seed yields identical study->split assignment."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp)

        CreateSplits(_make_ctx(tmp)).transform([])
        first = {r["study_id"]: r["split"] for r in _read_records(jsonl_path)}

        CreateSplits(_make_ctx(tmp)).transform([])
        second = {r["study_id"]: r["split"] for r in _read_records(jsonl_path)}

        assert first == second


def test_split_sizes_roughly_match_ratios():
    """Study-level split sizes should roughly follow split_ratios (0.7/0.15/0.15)."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp, num_studies=40)
        CreateSplits(_make_ctx(tmp)).transform([])

        mapping = _study_to_split(_read_records(jsonl_path))
        split_of_study = {study: next(iter(splits)) for study, splits in mapping.items()}
        total = len(split_of_study)
        counts = {
            "train": sum(1 for s in split_of_study.values() if s == "train"),
            "val": sum(1 for s in split_of_study.values() if s == "val"),
            "test": sum(1 for s in split_of_study.values() if s == "test"),
        }
        assert sum(counts.values()) == total
        # Allow a generous tolerance for rounding within strata.
        assert abs(counts["train"] / total - 0.7) < 0.15
        assert abs(counts["val"] / total - 0.15) < 0.15
        assert abs(counts["test"] / total - 0.15) < 0.15


def test_manifest_only_leaves_jsonl_untouched():
    """With split_manifest_only, no `split` field is added and a manifest is written."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = _write_jsonl(tmp, num_studies=10)
        ctx = _make_ctx(tmp)
        ctx.metadata.split_manifest_only = True
        CreateSplits(ctx).transform([])

        assert all("split" not in r for r in _read_records(jsonl_path))
        manifest = os.path.join(tmp, "99_synthetic", "reports", "split_manifest.json")
        assert os.path.exists(manifest)
