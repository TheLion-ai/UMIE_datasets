"""Opt-in step that rewrites the dataset JSONL from the flat v1 schema to the hierarchical v2 schema.

Part of Theme K (Task 33). It is a **no-op unless** the run opts in via ``PathArgs.schema_version
== "2.0"``, so v1 output stays byte-identical by default. When enabled it runs last (appended
automatically by ``BasePipeline.pipeline``), after the JSONL has been fully populated with labels and
provenance, and converts every record in place using :func:`metadata_schema.build_v2_record`.

It enriches the ``png_representation`` with the *actual* intensity transform applied during 2D
conversion (the DICOM window from ``DicomConfig``), so the v2 record faithfully records how PNG
pixel values were derived.
"""

import jsonlines

from base.step import BaseStep
from constants import SCHEMA_VERSION_V2
from metadata_schema import build_png_representation, build_v2_record


class ConvertJsonlToV2(BaseStep):
    """Rewrite the dataset JSONL in the v2 hierarchical schema when ``schema_version == '2.0'``."""

    def transform(self, X: list) -> list:
        """Convert the dataset JSONL to v2 in place (or do nothing when v2 is not opted in).

        Args:
            X (list): Image paths flowing through the pipeline (returned unchanged).

        Returns:
            list: ``X`` unchanged - this step only rewrites the JSONL sidecar.
        """
        if self.schema_version != SCHEMA_VERSION_V2:
            return X

        with jsonlines.open(self.json_path, mode="r") as reader:
            v1_records = list(reader)

        v2_records = [self._to_v2(record) for record in v1_records]

        with jsonlines.open(self.json_path, mode="w") as writer:
            for record in v2_records:
                writer.write(record)
        return X

    def _to_v2(self, record: dict) -> dict:
        """Convert a single v1 record to v2, recording the actual PNG intensity transform."""
        png_representation = None
        is_volume = record.get("umie_path", "").endswith(".nii.gz") or bool(record.get("volume_metadata"))
        if not is_volume and (self.window_center is not None or self.window_width is not None):
            png_representation = build_png_representation(
                type="native_2d",
                single_image_path=record.get("umie_path", ""),
                single_mask_path=record.get("mask_path") or None,
                window_center=self.window_center,
                window_width=self.window_width,
            )
        return build_v2_record(record, png_representation=png_representation)
