"""Unit tests for ExportHuggingFace: on-disk HuggingFace ``datasets`` export + dataset card."""

import json
import os
import tempfile

import jsonlines
import numpy as np
from PIL import Image as PILImage

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from config.provenance import get_provenance
from datasets import DatasetDict, load_from_disk  # type: ignore[attr-defined]
from src.steps.export_huggingface import ExportHuggingFace

DATASET_NAME = "kits23"
DATASET_UID = "99"
DATASET_DIR = f"{DATASET_UID}_{DATASET_NAME}"


def _make_ctx(tmp: str, dataset_name: str = DATASET_NAME) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid=DATASET_UID, dataset_name=dataset_name, modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_png(path: str) -> None:
    """Write a small real RGB PNG at ``path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    PILImage.fromarray(array).save(path)


def _build_tiny_dataset(tmp: str) -> list:
    """Create a tiny dataset dir with PNGs + a split-aware JSONL; return the written records."""
    records = []
    splits = ["train", "train", "val", "test"]
    for index, split in enumerate(splits):
        umie_id = f"99_0_{index:03d}_x.png"
        umie_path = f"{DATASET_DIR}/CT/Images/{umie_id}"
        mask_path = f"{DATASET_DIR}/CT/Masks/{umie_id}"
        _write_png(os.path.join(tmp, umie_path))
        _write_png(os.path.join(tmp, mask_path))
        records.append(
            {
                "umie_path": umie_path,
                "dataset_name": DATASET_NAME,
                "dataset_uid": DATASET_UID,
                "modality_name": "CT",
                "comparative": "",
                "study_id": str(index),
                "umie_id": umie_id,
                "mask_path": mask_path,
                "labels": [{"normal": 1}, {"neoplasm": 1}],
                "source_labels": ["normal"],
                "split": split,
            }
        )

    jsonl_path = os.path.join(tmp, DATASET_DIR, f"{DATASET_DIR}.jsonl")
    with jsonlines.open(jsonl_path, mode="w") as writer:
        for record in records:
            writer.write(record)
    return records


def test_export_roundtrips_with_splits_and_card():
    """Run the step, then load_from_disk and assert splits, counts, labels and the card."""
    with tempfile.TemporaryDirectory() as tmp:
        records = _build_tiny_dataset(tmp)
        ctx = _make_ctx(tmp)
        export_dir = os.path.join(tmp, "hf_export")
        ctx.export.hf_export_path = export_dir

        result = ExportHuggingFace(ctx).transform(["unchanged"])
        # (transform returns X unchanged)
        assert result == ["unchanged"]

        loaded = load_from_disk(export_dir)

        # (a) DatasetDict with the expected splits.
        assert isinstance(loaded, DatasetDict)
        assert set(loaded.keys()) == {"train", "val", "test"}
        assert loaded["train"].num_rows == 2
        assert loaded["val"].num_rows == 1
        assert loaded["test"].num_rows == 1

        # (b) Total record count matches the JSONL.
        total = sum(split.num_rows for split in loaded.values())
        assert total == len(records)

        # (c) The labels column round-trips (flattened JSON string -> {name: grade}).
        labels = json.loads(loaded["train"][0]["labels"])
        assert labels == {"normal": 1, "neoplasm": 1}

        # Image column round-trips to a real PIL image.
        assert loaded["train"][0]["image"].size == (8, 8)
        # Scalar fields are preserved.
        assert loaded["train"][0]["dataset_uid"] == DATASET_UID

        # (d) README.md card exists and contains the provenance license string.
        card_path = os.path.join(export_dir, "README.md")
        assert os.path.exists(card_path)
        with open(card_path, encoding="utf-8") as card:
            card_text = card.read()
        assert get_provenance(DATASET_NAME).license in card_text
        assert "CT" in card_text


def test_export_defaults_path_and_single_split():
    """Without a split field the export is a single-split DatasetDict at the default path."""
    with tempfile.TemporaryDirectory() as tmp:
        records = _build_tiny_dataset(tmp)
        # Strip the split field and rewrite the JSONL.
        jsonl_path = os.path.join(tmp, DATASET_DIR, f"{DATASET_DIR}.jsonl")
        for record in records:
            record.pop("split", None)
        with jsonlines.open(jsonl_path, mode="w") as writer:
            for record in records:
                writer.write(record)

        ctx = _make_ctx(tmp)
        ExportHuggingFace(ctx).transform([])

        # Default export path is f"{dataset_root}/huggingface".
        export_dir = os.path.join(tmp, DATASET_DIR, "huggingface")
        loaded = load_from_disk(export_dir)
        assert isinstance(loaded, DatasetDict)
        assert set(loaded.keys()) == {"train"}
        assert loaded["train"].num_rows == len(records)
