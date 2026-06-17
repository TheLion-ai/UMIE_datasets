"""Export the processed dataset to the HuggingFace ``datasets`` Arrow format on disk (Task 30).

This optional, opt-in pipeline step reads the finished dataset JSONL produced by the rest of
the pipeline and materializes it as a HuggingFace ``datasets`` dataset on local disk via
``DatasetDict.save_to_disk`` -- there is no network push (publishing to the Hub stays in
``utils/huggingface_cli.py``). The export is purely additive: it only *reads* the processed
outputs (the JSONL and the referenced PNG images/masks) and *writes* a self-contained
HuggingFace dataset directory plus a dataset card, never touching the UMIE ids, folder layout,
or the existing image / JSONL outputs.

The on-disk schema mirrors the push-based logic in ``utils/huggingface_cli.py``: the image
(and mask, when present) columns are cast with ``datasets.Image()``, ``labels`` is flattened to
a ``{name: grade}`` dict and stored as a JSON string, and ``dataset_uid`` / ``study_id`` are
kept as strings. When a ``split`` field is present the records are grouped into a
``DatasetDict`` with ``train`` / ``val`` / ``test`` splits; otherwise everything is placed in a
single ``train`` split so the output is always a ``DatasetDict`` and therefore consistently
round-trip-able with ``datasets.load_from_disk``.

Future work: additional first-class exporters for medical / training-oriented formats are
planned but intentionally out of scope here -- MONAI / nnU-Net dataset layouts for segmentation
training, and sharded ``WebDataset`` / ``TFRecord`` exports for large-scale streaming.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from base.step import BaseStep
from config.provenance import get_provenance
from datasets import (  # type: ignore[attr-defined]
    Dataset,
    DatasetDict,
    Features,
    Image,
    Value,
)

# Canonical ordering of splits when a "split" field is present in the JSONL (Task 21).
SPLIT_ORDER = ("train", "val", "test")
# Fallback single-split name used when the records carry no "split" field.
DEFAULT_SPLIT = "train"


class ExportHuggingFace(BaseStep):
    """Write the processed dataset to a local HuggingFace Arrow dataset + card (Task 30)."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Materialize the dataset JSONL as a HuggingFace ``datasets`` dataset on disk.

        Reads ``self.json_path``, builds absolute image (and mask) paths from
        ``self.target_path``, casts the schema to mirror ``utils/huggingface_cli`` and writes a
        ``DatasetDict`` (split-aware when a ``split`` field is present) to the configured export
        directory together with an auto-generated ``README.md`` dataset card. ``X`` is returned
        unchanged; the step is a no-op when the source JSONL does not exist.

        Args:
            X (list): List of paths to the images (unchanged on return).

        Returns:
            list: The unchanged list of image paths.
        """
        if not os.path.exists(self.json_path):
            print(f"HuggingFace export skipped: dataset JSONL not found at {self.json_path}.")
            return X

        export_dir = self.export_config.hf_export_path or os.path.join(self.dataset_root, "huggingface")

        records = self._read_records()
        if not records:
            print("HuggingFace export skipped: dataset JSONL is empty.")
            return X

        has_masks = any(bool(record.get("mask_path")) for record in records)
        features = self._build_features(records[0], has_masks)

        print(f"Exporting {len(records)} record(s) to HuggingFace dataset at {export_dir} ...")
        dataset_dict = self._build_dataset_dict(records, features, has_masks)

        os.makedirs(export_dir, exist_ok=True)
        dataset_dict.save_to_disk(export_dir)
        self._write_card(export_dir, records, dataset_dict)

        print(f"HuggingFace export complete: {sum(d.num_rows for d in dataset_dict.values())} record(s) written.")
        return X

    def _read_records(self) -> list[dict[str, Any]]:
        """Read the dataset JSONL into a list of plain dict records via pandas.

        Returns:
            list[dict[str, Any]]: One dict per JSONL line, in file order.
        """
        df = pd.read_json(self.json_path, lines=True)
        return df.to_dict(orient="records")

    @staticmethod
    def _flatten_labels(labels: Any) -> str:
        """Flatten a list of ``{name: grade}`` dicts into one ``{name: grade}`` JSON string.

        Mirrors ``utils/huggingface_cli.transform_labels`` so the on-disk schema matches the
        push-based export. Accepts either an already-decoded list or a JSON string.

        Args:
            labels (Any): The raw ``labels`` value from a record (list of dicts or JSON string).

        Returns:
            str: A JSON string of the merged ``{name: grade}`` mapping.
        """
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON format: {labels}") from exc

        merged: dict[str, Any] = {}
        for label in labels or []:
            if isinstance(label, dict):
                for key, value in label.items():
                    if value is not None:
                        merged[key] = value
            else:
                raise ValueError(f"Invalid label type: {type(label)}")
        return json.dumps(merged)

    def _build_features(self, sample_record: dict[str, Any], has_masks: bool) -> Features:
        """Build the HuggingFace ``Features`` schema for the export.

        The ``image`` (and ``mask`` when masks exist) columns are cast with ``Image()``;
        ``labels`` is a JSON string; ``dataset_uid`` / ``study_id`` are strings; every other
        scalar field present in the sample record is kept as a string ``Value``.

        Args:
            sample_record (dict[str, Any]): A representative record used to discover columns.
            has_masks (bool): Whether the dataset has masks (adds a ``mask`` Image column).

        Returns:
            Features: The HuggingFace feature schema for the exported dataset.
        """
        feature_map: dict[str, Any] = {"image": Image(), "labels": Value("string")}
        if has_masks:
            feature_map["mask"] = Image()

        # Keep every remaining scalar field (e.g. dataset_name, modality_name, comparative,
        # umie_id, split, license, source_dataset, source_labels) as a string Value. Keys that
        # become the image/mask columns are remapped from umie_path / mask_path below.
        skip = {"umie_path", "mask_path", "labels"}
        for key in sample_record:
            if key in skip:
                continue
            feature_map[key] = Value("string")
        return Features(feature_map)

    def _record_to_row(self, record: dict[str, Any], has_masks: bool) -> dict[str, Any]:
        """Convert one JSONL record into a HF row matching the ``_build_features`` schema.

        Args:
            record (dict[str, Any]): A single JSONL record.
            has_masks (bool): Whether the dataset has masks.

        Returns:
            dict[str, Any]: A row keyed by the HF feature names.
        """
        row: dict[str, Any] = {
            "image": os.path.join(self.target_path, str(record["umie_path"])),
            "labels": self._flatten_labels(record.get("labels")),
        }
        if has_masks:
            mask_path = record.get("mask_path")
            row["mask"] = os.path.join(self.target_path, str(mask_path)) if mask_path else None

        skip = {"umie_path", "mask_path", "labels"}
        for key, value in record.items():
            if key in skip:
                continue
            row[key] = "" if value is None else str(value)
        return row

    def _build_dataset_dict(
        self,
        records: list[dict[str, Any]],
        features: Features,
        has_masks: bool,
    ) -> DatasetDict:
        """Group records into a split-aware ``DatasetDict`` and cast to the feature schema.

        When the records carry a ``split`` field, the resulting ``DatasetDict`` contains the
        ``train`` / ``val`` / ``test`` splits that are actually present; otherwise all records
        go into a single ``train`` split. The output is always a ``DatasetDict``.

        Args:
            records (list[dict[str, Any]]): All JSONL records.
            features (Features): The feature schema to cast each split to.
            has_masks (bool): Whether the dataset has masks.

        Returns:
            DatasetDict: The split-aware dataset, image/mask columns cast with ``Image()``.
        """
        rows = [self._record_to_row(record, has_masks) for record in records]

        has_split = any(record.get("split") for record in records)
        grouped: dict[str, list[dict[str, Any]]] = {}
        if has_split:
            for record, row in zip(records, rows):
                split = str(record.get("split") or DEFAULT_SPLIT)
                grouped.setdefault(split, []).append(row)
        else:
            grouped[DEFAULT_SPLIT] = rows

        # Emit known splits in canonical order first, then any unexpected extras deterministically.
        ordered_splits = [s for s in SPLIT_ORDER if s in grouped]
        ordered_splits += sorted(s for s in grouped if s not in SPLIT_ORDER)

        splits = {split: Dataset.from_list(grouped[split], features=features) for split in ordered_splits}
        return DatasetDict(splits)

    def _modality_breakdown(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """Count records per ``modality_name`` for the dataset card.

        Args:
            records (list[dict[str, Any]]): All JSONL records.

        Returns:
            dict[str, int]: Record count keyed by modality name (deterministically ordered).
        """
        counts: dict[str, int] = {}
        for record in records:
            modality = str(record.get("modality_name") or "unknown")
            counts[modality] = counts.get(modality, 0) + 1
        return dict(sorted(counts.items()))

    def _write_card(
        self,
        export_dir: str,
        records: list[dict[str, Any]],
        dataset_dict: DatasetDict,
    ) -> None:
        """Write a ``README.md`` dataset card describing license, provenance and breakdowns.

        Args:
            export_dir (str): Directory the dataset was saved to (card is written here).
            records (list[dict[str, Any]]): All JSONL records (for the breakdowns).
            dataset_dict (DatasetDict): The saved dataset (for per-split counts).
        """
        provenance = get_provenance(self.dataset_name)
        total = sum(split.num_rows for split in dataset_dict.values())

        lines: list[str] = [
            f"# {self.dataset_uid}_{self.dataset_name}",
            "",
            "HuggingFace `datasets` export generated by the UMIE pipeline "
            "(`ExportHuggingFace`, Task 30). Load it with "
            "`datasets.load_from_disk(<this directory>)`.",
            "",
            "## License & source attribution",
            "",
            f"- **License:** {provenance.license}",
            f"- **Source dataset:** {provenance.source_dataset}",
        ]
        if provenance.source_url:
            lines.append(f"- **Source URL:** {provenance.source_url}")
        if provenance.source_citation:
            lines.append(f"- **Citation:** {provenance.source_citation}")
        if provenance.redistributable is not None:
            lines.append(f"- **Redistributable:** {provenance.redistributable}")

        lines += [
            "",
            "## Record counts",
            "",
            f"- **Total records:** {total}",
        ]
        for split, dataset in dataset_dict.items():
            lines.append(f"- **{split}:** {dataset.num_rows}")

        lines += [
            "",
            "## Modality breakdown",
            "",
        ]
        for modality, count in self._modality_breakdown(records).items():
            lines.append(f"- **{modality}:** {count}")
        lines.append("")

        with open(os.path.join(export_dir, "README.md"), mode="w", encoding="utf-8") as card:
            card.write("\n".join(lines))
