"""Unit tests for the distribution_report utility (Task 22)."""

import os
import tempfile

import jsonlines

from utils.distribution_report import generate_distribution_report


def _write_dataset(data_dir: str, uid: str, name: str, records: list) -> None:
    """Write one dataset JSONL at {data_dir}/{uid}_{name}/{uid}_{name}.jsonl."""
    dataset_dir = os.path.join(data_dir, f"{uid}_{name}")
    os.makedirs(dataset_dir, exist_ok=True)
    jsonl_path = os.path.join(dataset_dir, f"{uid}_{name}.jsonl")
    with jsonlines.open(jsonl_path, mode="w") as writer:
        for record in records:
            writer.write(record)


def _record(study_id: str, phase_name: str, labels: list) -> dict:
    """Build a minimal JSONL record."""
    return {
        "umie_path": f"x/{study_id}.png",
        "dataset_name": "ds",
        "dataset_uid": "00",
        "phase_name": phase_name,
        "comparative": "",
        "study_id": study_id,
        "umie_id": f"{study_id}.png",
        "mask_path": "",
        "labels": labels,
        "source_labels": [],
    }


def test_label_and_modality_counts():
    """The returned dict has the expected per-dataset and global label / modality counts."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_dataset(
            tmp,
            "00",
            "alpha",
            [
                _record("1", "CT", [{"Neoplasm": 1}]),
                _record("2", "CT", [{"Neoplasm": 1}, {"NormalityDecriptor": 1}]),
                _record("3", "MRI", []),
            ],
        )
        _write_dataset(
            tmp,
            "01",
            "beta",
            [
                _record("1", "XRay", [{"NormalityDecriptor": 1}]),
            ],
        )

        stats = generate_distribution_report(tmp)

        assert stats["num_datasets"] == 2
        assert stats["total_records"] == 4

        alpha = stats["per_dataset"]["00_alpha"]
        assert alpha["label_counts"] == {"Neoplasm": 2, "NormalityDecriptor": 1}
        assert alpha["modality_counts"] == {"CT": 2, "MRI": 1}
        # class imbalance = max(2)/min(1) = 2.0
        assert alpha["class_imbalance_ratio"] == 2.0

        beta = stats["per_dataset"]["01_beta"]
        assert beta["label_counts"] == {"NormalityDecriptor": 1}
        assert beta["modality_counts"] == {"XRay": 1}
        # single label -> no imbalance ratio
        assert beta["class_imbalance_ratio"] is None

        glob_labels = stats["global"]["label_counts"]
        assert glob_labels == {"Neoplasm": 2, "NormalityDecriptor": 2}
        assert stats["global"]["modality_counts"] == {"CT": 2, "MRI": 1, "XRay": 1}


def test_reports_written_and_empty_dir_is_safe():
    """Markdown + CSV reports are written by default; an empty dir produces empty stats."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_dataset(tmp, "00", "alpha", [_record("1", "CT", [{"Neoplasm": 1}])])
        generate_distribution_report(tmp)
        assert os.path.exists(os.path.join(tmp, "distribution_report.md"))
        assert os.path.exists(os.path.join(tmp, "distribution_report.csv"))

    with tempfile.TemporaryDirectory() as empty:
        stats = generate_distribution_report(empty)
        assert stats["num_datasets"] == 0
        assert stats["total_records"] == 0
        assert stats["global"]["label_counts"] == {}
