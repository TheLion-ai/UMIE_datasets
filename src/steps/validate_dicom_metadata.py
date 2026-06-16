"""Validate required DICOM metadata tags on source files before conversion."""

import json
import os

import pydicom

from base.step import BaseStep


class ValidateDicomMetadata(BaseStep):
    """Validate required DICOM metadata tags on source files before conversion."""

    def transform(
        self,
        X: list,  # source file paths
    ) -> list:
        """Check source DICOM files against ``quality.required_dicom_tags`` and report issues.

        Each ``.dcm`` path in ``X`` is read with pydicom and validated against
        ``quality.required_dicom_tags`` (``{tag_name: expected_value_or_None}``; ``None`` means
        the tag must merely be present, a value means it must equal). Files with missing or
        unexpected tags are written to a JSON report. When ``quality.exclude_invalid_dicom`` is
        set, invalid files are dropped from the returned list; otherwise ``X`` is returned
        unchanged. UMIE ids are never altered.

        Args:
            X (list): List of source file paths.
        Returns:
            list: ``X`` unchanged, or with invalid DICOM files removed when configured to drop.
        """
        required = self.quality.required_dicom_tags
        if not required:
            return X

        print("Validating DICOM metadata...")
        invalid: dict = {}
        for path in X:
            if not str(path).endswith(".dcm"):
                continue
            issues = self._validate_file(path, required)
            if issues:
                invalid[self._report_key(path)] = issues

        self._write_report(invalid, required)
        print(f"DICOM metadata validation complete: {len(invalid)} invalid file(s).")

        if self.quality.exclude_invalid_dicom and invalid:
            invalid_keys = set(invalid.keys())
            return [path for path in X if self._report_key(path) not in invalid_keys]
        return X

    def _report_key(self, path: str) -> str:
        """Return a stable identifier for a source path used as a report key.

        Args:
            path (str): Source file path.
        Returns:
            str: Path relative to the source dir when possible, else the original path.
        """
        try:
            return os.path.relpath(path, self.source_path)
        except ValueError:
            return path

    def _validate_file(self, path: str, required: dict) -> list:
        """Validate one DICOM file and return a list of issue descriptions.

        Args:
            path (str): Path to the ``.dcm`` file.
            required (dict): ``{tag_name: expected_value_or_None}`` requirements.
        Returns:
            list: Issue dicts, empty when the file satisfies every requirement.
        """
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as error:
            return [{"tag": None, "issue": "unreadable", "detail": str(error)}]

        issues = []
        for tag_name, expected in required.items():
            if tag_name not in dataset:
                issues.append({"tag": tag_name, "issue": "missing"})
                continue
            if expected is not None:
                actual = dataset.get(tag_name)
                if str(actual) != str(expected):
                    issues.append(
                        {"tag": tag_name, "issue": "unexpected_value", "expected": expected, "actual": str(actual)}
                    )
        return issues

    def _write_report(self, invalid: dict, required: dict) -> None:
        """Write the DICOM metadata report to JSON under the reports dir.

        Args:
            invalid (dict): Mapping of report key to its list of issues.
            required (dict): The required-tag configuration that was checked.
        """
        report = {
            "required_dicom_tags": {key: value for key, value in required.items()},
            "exclude_invalid_dicom": self.quality.exclude_invalid_dicom,
            "invalid": invalid,
        }
        report_path = os.path.join(self.reports_dir(), "dicom_metadata_report.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)
