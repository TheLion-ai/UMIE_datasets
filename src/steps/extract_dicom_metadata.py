"""Extract a configurable, de-identified set of DICOM header tags into the JSONL (Task 20).

This optional, opt-in step enriches each dataset JSONL record with a small set of acquisition
/ technical DICOM tags (never pixel data). It is additive: existing JSONL fields are never
changed, only new keys are added (or, when ``metadata_config.metadata_sidecar`` is set, the
extracted tags are written to a sidecar JSON under ``reports_dir()/dicom_metadata.json`` and
the JSONL is left untouched).

Which tags are stored
---------------------
The set of tags is configured per pipeline through ``metadata_config.dicom_tags`` (a list of
DICOM keyword strings, e.g. ``["PatientAge", "PatientSex", "SliceThickness", "KVP",
"XRayTubeCurrent", "Manufacturer", "ManufacturerModelName"]``). When ``dicom_tags`` is ``None``
the step is a no-op and writes nothing.

De-identification (default ON, ``metadata_config.deidentify``)
--------------------------------------------------------------
A module-level PHI denylist (``PHI_DENYLIST``) lists DICOM keywords that directly identify a
patient (name, id, addresses, etc.). When ``deidentify`` is ``True`` (the default), these tags
are NEVER written even if they are explicitly requested in ``dicom_tags``.

Dates carry indirect PHI (they can re-identify a patient when combined with other records), so
any requested DICOM *date* tag (keyword ending in ``Date``) is not written verbatim while
de-identifying. Instead we apply a deterministic, per-study ``study_date_offset_days`` shift:
the original date is parsed and shifted by a fixed, study-derived number of days, and only the
resulting shifted date is stored (alongside the offset itself under
``<TagName>_offset_days``) so relative timing between a patient's studies is preserved while the
true calendar date is hidden. The offset is derived deterministically from the record's
``study_id`` so the same study always shifts by the same amount and the shift is reproducible.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import jsonlines
import pydicom

from base.step import BaseStep

# Module-level PHI denylist: DICOM keywords that directly identify a patient. When
# de-identification is on (the default) these are NEVER written, even if explicitly requested.
PHI_DENYLIST: frozenset[str] = frozenset(
    {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientBirthTime",
        "PatientAddress",
        "OtherPatientIDs",
        "OtherPatientNames",
        "OtherPatientIDsSequence",
        "PatientTelephoneNumbers",
        "PatientMotherBirthName",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "InstitutionName",
        "InstitutionAddress",
        "IssuerOfPatientID",
        "MilitaryRank",
        "EthnicGroup",
        "PatientReligiousPreference",
        "PatientInsurancePlanCodeSequence",
        "DeviceSerialNumber",
        "AccessionNumber",
    }
)

# How many days to clamp a deterministic per-study date shift to (kept small so relative
# ordering between a patient's studies is preserved while the true calendar date is hidden).
_MAX_DATE_OFFSET_DAYS: int = 30


class ExtractDicomMetadata(BaseStep):
    """Extract configured, de-identified DICOM header tags into additive JSONL fields (Task 20)."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Enrich the dataset JSONL (or a sidecar) with the configured DICOM tags.

        No-op when ``metadata_config.dicom_tags`` is ``None``. Each JSONL record is mapped to a
        source ``.dcm`` via ``source_paths.json`` if present, else by matching basename against
        ``X``. Requested non-PHI tags are written; PHI tags are dropped and date tags are shifted
        when ``deidentify`` is set. ``X`` is returned unchanged.

        Args:
            X (list): List of paths to the images (unchanged on return).
        Returns:
            list: The unchanged list of image paths.
        """
        tags = self.metadata_config.dicom_tags
        if not tags:
            return X

        print("Extracting DICOM metadata...")
        source_paths = self._load_source_paths()
        basename_index = self._index_by_basename(X)

        if not os.path.exists(self.json_path):
            return X

        records: list[dict] = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                records.append(obj)

        extracted: dict[str, dict[str, Any]] = {}
        for obj in records:
            dcm_path = self._resolve_dcm_path(obj, source_paths, basename_index)
            if dcm_path is None:
                continue
            metadata = self._extract_tags(dcm_path, list(tags), str(obj.get("study_id", "")))
            if metadata:
                extracted[obj["umie_path"]] = metadata

        if self.metadata_config.metadata_sidecar:
            self._write_sidecar(extracted)
        else:
            self._enrich_jsonl(records, extracted)

        print(f"DICOM metadata extraction complete: enriched {len(extracted)} record(s).")
        return X

    def _load_source_paths(self) -> Optional[dict]:
        """Load ``source_paths.json`` (``{umie_output_path: source_file_path}``) if present.

        Returns:
            Optional[dict]: The mapping, or ``None`` when the file does not exist.
        """
        path = os.path.join(self.target_path, "source_paths.json")
        if os.path.exists(path):
            with open(path) as handle:
                return json.load(handle)
        return None

    def _index_by_basename(self, X: list) -> dict[str, str]:
        """Build a ``{basename: path}`` index of source ``.dcm`` files in ``X``.

        Args:
            X (list): List of source/image paths.
        Returns:
            dict[str, str]: Mapping of file basename to the first matching path.
        """
        index: dict[str, str] = {}
        for path in X:
            base = os.path.basename(str(path))
            index.setdefault(base, str(path))
        return index

    def _resolve_dcm_path(
        self,
        record: dict,
        source_paths: Optional[dict],
        basename_index: dict[str, str],
    ) -> Optional[str]:
        """Resolve a JSONL record to its source ``.dcm`` path.

        Prefers ``source_paths.json`` (keyed by the absolute umie output path); falls back to
        matching the record's ``umie_id`` basename against the source-file index.

        Args:
            record (dict): One JSONL record.
            source_paths (Optional[dict]): The ``source_paths.json`` mapping, if loaded.
            basename_index (dict[str, str]): ``{basename: source_path}`` fallback index.
        Returns:
            Optional[str]: An existing ``.dcm`` path, or ``None`` when none can be resolved.
        """
        if source_paths:
            umie_abs = os.path.join(self.target_path, record["umie_path"])
            candidate = source_paths.get(umie_abs) or source_paths.get(record["umie_path"])
            if candidate and str(candidate).endswith(".dcm") and os.path.exists(candidate):
                return str(candidate)

        umie_id = record.get("umie_id", "")
        base = os.path.splitext(umie_id)[0]
        for key in (umie_id, f"{base}.dcm", base):
            candidate = basename_index.get(key)
            if candidate and str(candidate).endswith(".dcm") and os.path.exists(candidate):
                return candidate
        return None

    def _extract_tags(self, dcm_path: str, tags: list[str], study_id: str) -> dict[str, Any]:
        """Read the requested tags from one DICOM file, applying de-identification rules.

        Args:
            dcm_path (str): Path to the source ``.dcm`` file.
            tags (list[str]): Requested DICOM keyword strings.
            study_id (str): The record's study id, used to derive a deterministic date offset.
        Returns:
            dict[str, Any]: Mapping of stored tag name to value (PHI tags omitted).
        """
        try:
            dataset = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        except Exception:
            return {}

        deidentify = self.metadata_config.deidentify
        result: dict[str, Any] = {}
        for tag in tags:
            if deidentify and tag in PHI_DENYLIST:
                # Never write a denylisted tag when de-identifying, even if requested.
                continue
            if tag not in dataset:
                continue
            value = dataset.get(tag)
            if value is None:
                continue
            if tag.endswith("Date") and deidentify:
                shifted = self._shift_date(str(value), study_id)
                if shifted is not None:
                    result[tag] = shifted
                    result[f"{tag}_offset_days"] = self._date_offset_days(study_id)
                continue
            result[tag] = self._to_jsonable(value)
        return result

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Coerce a pydicom value into a JSON-serialisable scalar/list.

        Args:
            value (Any): Raw pydicom element value.
        Returns:
            Any: A JSON-serialisable representation (str, number, bool, or list thereof).
        """
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            return [ExtractDicomMetadata._to_jsonable(item) for item in value]
        except TypeError:
            return str(value)

    @staticmethod
    def _date_offset_days(study_id: str) -> int:
        """Return a deterministic per-study date offset in ``[-MAX, +MAX]`` days.

        Derived from the study id so the same study always shifts by the same amount (relative
        timing between a patient's studies is preserved) and the shift is reproducible.

        Args:
            study_id (str): The record's study id.
        Returns:
            int: The signed day offset.
        """
        digest = sum(ord(char) for char in study_id) if study_id else 0
        span = 2 * _MAX_DATE_OFFSET_DAYS + 1
        return (digest % span) - _MAX_DATE_OFFSET_DAYS

    def _shift_date(self, raw: str, study_id: str) -> Optional[str]:
        """Shift a ``YYYYMMDD`` DICOM date by the study's deterministic offset.

        Args:
            raw (str): The raw DICOM date value.
            study_id (str): The record's study id (drives the offset).
        Returns:
            Optional[str]: The shifted ``YYYYMMDD`` date, or ``None`` if ``raw`` is unparseable.
        """
        raw = raw.strip()
        try:
            parsed = datetime.strptime(raw, "%Y%m%d")
        except ValueError:
            return None
        shifted = parsed + timedelta(days=self._date_offset_days(study_id))
        return shifted.strftime("%Y%m%d")

    def _enrich_jsonl(self, records: list[dict], extracted: dict[str, dict[str, Any]]) -> None:
        """Rewrite the JSONL in place, adding a ``dicom_metadata`` field to matching records.

        Existing keys and record order are preserved; only the additive ``dicom_metadata`` key
        is added (records without extracted metadata are written back unchanged).

        Args:
            records (list[dict]): All JSONL records, in original order.
            extracted (dict[str, dict[str, Any]]): ``{umie_path: metadata}`` mapping.
        """
        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in records:
                metadata = extracted.get(obj["umie_path"])
                if metadata:
                    obj["dicom_metadata"] = metadata
                writer.write(obj)

    def _write_sidecar(self, extracted: dict[str, dict[str, Any]]) -> None:
        """Write the extracted metadata to a sidecar JSON under the reports dir.

        Args:
            extracted (dict[str, dict[str, Any]]): ``{umie_path: metadata}`` mapping.
        """
        path = os.path.join(self.reports_dir(), "dicom_metadata.json")
        with open(path, "w") as handle:
            json.dump(extracted, handle, indent=2)
