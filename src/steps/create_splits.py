"""Generate reproducible study-level stratified train/val/test splits (Task 21).

This optional, opt-in step assigns every JSONL record to a ``train`` / ``val`` / ``test`` split.
The split is keyed on ``study_id`` (NOT on the individual image), so every record sharing a
study id always lands in the same split. This is critical for 3D volumes, where many slices
share one study and an image-level split would leak slices of the same study across splits.

Determinism
-----------
Assignment uses ``random.Random(metadata_config.split_seed)`` (a local, seeded RNG; the global
``random`` state is never touched), so running twice with the same seed yields identical
assignments. Ratios come from ``metadata_config.split_ratios`` (train, val, test).

Stratification
--------------
When ``metadata_config.stratify_by_label`` is set, studies are grouped by their dominant label
(the most frequent RadLex label across the study's records, or ``"__none__"`` when unlabelled)
and the ratios are applied within each group, keeping the label distribution similar across
splits.

Output
------
Additive only: a ``split`` field is added to each JSONL record (existing keys and order are
preserved). When ``metadata_config.split_manifest_only`` is set, the JSONL is left untouched and
a ``reports_dir()/split_manifest.json`` mapping ``{study_id: split}`` is written instead.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

import jsonlines

from base.step import BaseStep


class CreateSplits(BaseStep):
    """Assign reproducible study-level stratified train/val/test splits (Task 21)."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Compute study-level splits and write them to the JSONL (or a manifest).

        Args:
            X (list): List of paths to the images (unchanged on return).
        Returns:
            list: The unchanged list of image paths.
        """
        if not os.path.exists(self.json_path):
            return X

        print("Creating reproducible study-level splits...")
        records: list[dict] = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                records.append(obj)

        if not records:
            return X

        study_to_split = self._assign_splits(records)

        if self.metadata_config.split_manifest_only:
            self._write_manifest(study_to_split)
        else:
            self._enrich_jsonl(records, study_to_split)

        counts = Counter(study_to_split.values())
        print(f"Split assignment complete: {dict(counts)} across {len(study_to_split)} studies.")
        return X

    def _assign_splits(self, records: list[dict]) -> dict[str, str]:
        """Assign each distinct ``study_id`` to a single split.

        Args:
            records (list[dict]): All JSONL records.
        Returns:
            dict[str, str]: Mapping of ``study_id`` to ``"train"`` / ``"val"`` / ``"test"``.
        """
        study_groups = self._group_studies(records)
        rng = random.Random(self.metadata_config.split_seed)

        study_to_split: dict[str, str] = {}
        # Iterate strata in a deterministic order (sorted by group key) so seeding is stable.
        for group_key in sorted(study_groups.keys()):
            studies = sorted(study_groups[group_key])
            for study_id, split in self._split_group(studies, rng).items():
                study_to_split[study_id] = split
        return study_to_split

    def _group_studies(self, records: list[dict]) -> dict[str, list[str]]:
        """Group distinct study ids by their stratification key (dominant label).

        When ``stratify_by_label`` is off, every study lands in a single ``"__all__"`` group.

        Args:
            records (list[dict]): All JSONL records.
        Returns:
            dict[str, list[str]]: ``{group_key: [study_id, ...]}`` with distinct study ids.
        """
        if not self.metadata_config.stratify_by_label:
            studies = sorted({str(obj.get("study_id", "")) for obj in records})
            return {"__all__": studies}

        # Accumulate label votes per study across all its records, then pick the dominant label.
        votes: dict[str, Counter] = defaultdict(Counter)
        for obj in records:
            study_id = str(obj.get("study_id", ""))
            for radlex_name in self._record_labels(obj):
                votes[study_id][radlex_name] += 1
            # Ensure unlabelled studies still appear (with no votes).
            votes.setdefault(study_id, Counter())

        groups: dict[str, list[str]] = defaultdict(list)
        for study_id, counter in votes.items():
            dominant = counter.most_common(1)[0][0] if counter else "__none__"
            groups[dominant].append(study_id)
        return groups

    @staticmethod
    def _record_labels(record: dict) -> list[str]:
        """Extract the RadLex label names from a record's ``labels`` list.

        ``labels`` is a list of ``{radlex_name: grade}`` dicts; this returns the radlex names.

        Args:
            record (dict): One JSONL record.
        Returns:
            list[str]: RadLex label names present on the record (possibly empty).
        """
        names: list[str] = []
        for label in record.get("labels", []) or []:
            if isinstance(label, dict):
                names.extend(str(key) for key in label.keys())
        return names

    def _split_group(self, studies: list[str], rng: random.Random) -> dict[str, str]:
        """Split one ordered group of study ids into train/val/test by the configured ratios.

        Uses the shared seeded RNG to shuffle, so assignment is reproducible across runs and the
        split sizes follow ``split_ratios``.

        Args:
            studies (list[str]): Distinct study ids in this stratum (pre-sorted).
            rng (random.Random): The shared seeded RNG.
        Returns:
            dict[str, str]: ``{study_id: split}`` for this group.
        """
        shuffled = list(studies)
        rng.shuffle(shuffled)

        train_ratio, val_ratio, _test_ratio = self.metadata_config.split_ratios
        total = len(shuffled)
        n_train = int(round(total * train_ratio))
        n_val = int(round(total * val_ratio))
        # Guard against rounding overflow; test gets whatever remains.
        n_train = min(n_train, total)
        n_val = min(n_val, total - n_train)

        assignment: dict[str, str] = {}
        for index, study_id in enumerate(shuffled):
            if index < n_train:
                assignment[study_id] = "train"
            elif index < n_train + n_val:
                assignment[study_id] = "val"
            else:
                assignment[study_id] = "test"
        return assignment

    def _enrich_jsonl(self, records: list[dict], study_to_split: dict[str, str]) -> None:
        """Rewrite the JSONL in place, adding an additive ``split`` field to each record.

        Existing keys and record order are preserved.

        Args:
            records (list[dict]): All JSONL records, in original order.
            study_to_split (dict[str, str]): ``{study_id: split}`` mapping.
        """
        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in records:
                study_id = str(obj.get("study_id", ""))
                obj["split"] = study_to_split.get(study_id, "train")
                writer.write(obj)

    def _write_manifest(self, study_to_split: dict[str, str]) -> None:
        """Write the ``{study_id: split}`` manifest to JSON under the reports dir.

        Args:
            study_to_split (dict[str, str]): ``{study_id: split}`` mapping.
        """
        manifest: dict[str, Any] = {
            "split_seed": self.metadata_config.split_seed,
            "split_ratios": list(self.metadata_config.split_ratios),
            "stratify_by_label": self.metadata_config.stratify_by_label,
            "study_to_split": study_to_split,
        }
        path = os.path.join(self.reports_dir(), "split_manifest.json")
        with open(path, "w") as handle:
            json.dump(manifest, handle, indent=2)
