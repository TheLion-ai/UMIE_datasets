"""Add additive license / source-attribution fields to every JSONL record (Task 23).

This optional, opt-in step stamps each dataset JSONL record with provenance metadata sourced
from the single canonical table in ``config/provenance.py`` (``get_provenance``). It is purely
additive: three new optional keys (``license``, ``source_dataset``, ``source_citation``) are
added to each record and no existing field is changed. Unknown datasets fall back to
``DEFAULT_PROVENANCE`` (``"unknown"``), so the fields are always populated.

The step is a no-op when ``metadata_config.add_provenance`` is ``False``.
"""

from __future__ import annotations

import os

import jsonlines

from base.step import BaseStep
from config.provenance import get_provenance


class AddProvenance(BaseStep):
    """Add additive ``license`` / ``source_dataset`` / ``source_citation`` JSONL fields (Task 23)."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Stamp every JSONL record with provenance from ``config/provenance.py``.

        No-op when ``metadata_config.add_provenance`` is ``False`` or the JSONL is missing.
        Existing fields and record order are preserved; only the additive provenance keys are
        added. ``X`` is returned unchanged.

        Args:
            X (list): List of paths to the images (unchanged on return).
        Returns:
            list: The unchanged list of image paths.
        """
        if not self.metadata_config.add_provenance:
            return X
        if not os.path.exists(self.json_path):
            return X

        print("Adding provenance (license / source attribution)...")
        provenance = get_provenance(self.dataset_name)

        records: list[dict] = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                records.append(obj)

        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in records:
                obj["license"] = provenance.license
                obj["source_dataset"] = provenance.source_dataset
                obj["source_citation"] = provenance.source_citation
                writer.write(obj)

        print(f"Provenance added to {len(records)} record(s): license={provenance.license!r}.")
        return X
