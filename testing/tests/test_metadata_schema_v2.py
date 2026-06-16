"""Tests for the versioned hierarchical JSONL schema v2 (Theme K: Tasks 33-35).

Covers the pure serializer (v1<->v2 round-trip, modality protocol vocab, png/nifti/segmentation
blocks) and the opt-in ``ConvertJsonlToV2`` pipeline step (no-op in v1, emits v2 when opted in).
"""

import os
import tempfile

import jsonlines
import pytest

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from constants import SCHEMA_VERSION_V1, SCHEMA_VERSION_V2
from metadata_schema import (
    build_imaging_protocol_details,
    build_nifti_files,
    build_png_representation,
    build_segmentations,
    build_v2_record,
    v2_to_v1,
)
from src.steps.convert_jsonl_to_v2 import ConvertJsonlToV2


def _v1_record(**overrides) -> dict:
    """A representative flat v1 record (as written by AddUmieIds + AddLabels)."""
    record = {
        "umie_path": "00_kits23/CT/Images/00_0_001_5.png",
        "dataset_name": "kits23",
        "dataset_uid": "00",
        "modality_name": "CT",
        "comparative": "",
        "study_id": "001",
        "umie_id": "00_0_001_5.png",
        "mask_path": "00_kits23/CT/Masks/00_0_001_5.png",
        "labels": [{"Neoplasm": 1}, {"Malignant": 1}],
        "source_labels": ["clear_cell_rcc"],
    }
    record.update(overrides)
    return record


# --- Task 33: v1 <-> v2 round-trip --------------------------------------------------------------


def test_v2_record_has_schema_version_and_hierarchy():
    """A v2 record carries schema_version 2.0 and the patient/study/series hierarchy."""
    v2 = build_v2_record(_v1_record())
    assert v2["schema_version"] == SCHEMA_VERSION_V2
    assert v2["entry_id"] == "00_0_001_5.png"
    assert v2["dataset_source_details"]["dataset_uid"] == "00"
    assert v2["series_level_info"]["modality"] == "CT"
    assert "patient_level_info" in v2 and "study_level_info" in v2


def test_round_trip_regenerates_v1_flat_fields():
    """v2_to_v1 regenerates umie_path, labels and source_labels (and the rest) exactly (Task 33 DOD)."""
    original = _v1_record()
    restored = v2_to_v1(build_v2_record(original))
    assert restored == original


def test_round_trip_for_volume_record():
    """A 3D volume record round-trips through nifti_files (umie_path recovered from image_path)."""
    original = _v1_record(
        umie_path="07_lits/CT/Images/07_0_003_2.nii.gz",
        mask_path="07_lits/CT/Masks/07_0_003_2.nii.gz",
        umie_id="07_0_003_2.nii.gz",
        volume_metadata={
            "shape": [512, 512, 75],
            "voxel_spacing_mm": [0.7, 0.7, 5.0],
            "orientation": "RAS",
            "affine": [[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 5.0, 0], [0, 0, 0, 1]],
        },
    )
    v2 = build_v2_record(original)
    assert v2["series_level_info"]["nifti_files"]["original_nifti_dimensions"] == [512, 512, 75]
    restored = v2_to_v1(v2)
    # volume_metadata is a v1 enrichment not carried in the flat fields; compare the flat subset.
    for key in ("umie_path", "mask_path", "umie_id", "labels", "source_labels", "modality_name"):
        assert restored[key] == original[key]


def test_v1_default_is_unchanged_emission():
    """Sanity: the v1 record is untouched by building v2 (no mutation, additive only)."""
    original = _v1_record()
    snapshot = dict(original)
    build_v2_record(original)
    assert original == snapshot


# --- Task 34: modality-specific protocol + controlled vocab -------------------------------------


def test_protocol_ct():
    """CT protocol fields populate; the contrast phase is validated against the vocab."""
    details = build_imaging_protocol_details(
        "CT", {"ct_contrast_phase": "Arterial", "ct_kvp": 120, "original_pixel_spacing_mm": [0.7, 0.7]}
    )
    assert details["ct_contrast_phase"] == "Arterial"
    assert details["ct_kvp"] == 120
    assert details["original_pixel_spacing_mm"] == [0.7, 0.7]


def test_protocol_mri_dual_coding():
    """An MRI sequence-type key term carries both _source and _radlex coding."""
    details = build_imaging_protocol_details("MRI", {"mri_sequence_type": "T1W", "mri_contrast_agent": "Gadolinium"})
    assert details["mri_sequence_type_source"] == "T1W"
    assert details["mri_sequence_type_radlex"]["id"] == "RID3897"
    assert details["mri_contrast_agent"] == "Gadolinium"


def test_protocol_xray_projection_dual_coding():
    """X-ray projection is a key term with dual coding."""
    details = build_imaging_protocol_details("Xray", {"xray_projection": "PA"})
    assert details["xray_projection_source"] == "PA"
    assert details["xray_projection_radlex"]["term"] == "posteroanterior projection"


def test_protocol_absent_fields_omitted_not_faked():
    """Fields the source lacks are simply absent, never invented."""
    details = build_imaging_protocol_details("CT", {"ct_contrast_phase": None, "ct_kvp": 100})
    assert "ct_contrast_phase" not in details
    assert details["ct_kvp"] == 100


def test_protocol_validation_rejects_unknown_vocab_value():
    """A controlled field with an out-of-vocabulary value raises (Task 34 'validated against the vocab')."""
    with pytest.raises(ValueError):
        build_imaging_protocol_details("Xray", {"xray_projection": "sideways"})


def test_protocol_fields_outside_modality_are_ignored():
    """Fields not valid for the modality are dropped."""
    details = build_imaging_protocol_details("Xray", {"ct_kvp": 120, "xray_projection": "AP"})
    assert "ct_kvp" not in details
    assert details["xray_projection_source"] == "AP"


# --- Task 35: png_representation / nifti_files / segmentations -----------------------------------


def test_png_representation_native_2d_records_normalization():
    """A native-2D png_representation records the actual windowing transform."""
    block = build_png_representation(
        type="native_2d",
        single_image_path="x/img.png",
        window_center=50,
        window_width=350,
        png_bit_depth=16,
    )
    assert block["type"] == "native_2d"
    assert block["png_window_center"] == 50
    assert block["normalization_applied_to_png"] == "Windowed (WC:50, WW:350), then scaled to 16-bit"


def test_png_representation_sliced_3d():
    """A sliced-3D png_representation carries the slice count and slicing axis."""
    block = build_png_representation(
        type="3d_sliced_to_2d",
        base_image_path_pattern="x/img_{:04d}.png",
        num_slices=75,
        slicing_axis=2,
    )
    assert block["num_slices"] == 75
    assert block["slicing_axis_of_original_volume"] == 2
    assert "Original intensities scaled to 8-bit" in block["normalization_applied_to_png"]


def test_png_representation_rejects_bad_type():
    """An unknown png_representation type is rejected."""
    with pytest.raises(ValueError):
        build_png_representation(type="weird")


def test_nifti_files_reuse_volume_metadata():
    """nifti_files is built from the Task 8 volume_metadata without recomputation."""
    volume_metadata = {
        "shape": [256, 256, 180],
        "voxel_spacing_mm": [1.0, 1.0, 1.0],
        "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    block = build_nifti_files(volume_metadata, image_path="x/vol.nii.gz", mask_path="x/mask.nii.gz")
    assert block["affine_matrix"] == volume_metadata["affine"]
    assert block["original_nifti_dimensions"] == [256, 256, 180]
    assert block["original_nifti_voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    assert block["mask_path"] == "x/mask.nii.gz"


def test_segmentations_dual_label_and_validation():
    """Segmentations carry dual labels + mask-value mapping + provenance, validated vs config/masks.py."""
    segs = build_segmentations(
        [{"structure": "Kidney", "structure_source": "kidney", "mask_value": 1, "provenance": "manual", "volume_cc": 120.5}]
    )
    assert segs[0]["structure_radlex"] == {"term": "Kidney", "id": "RID205"}
    assert segs[0]["mask_value_in_nifti"] == 1 and segs[0]["mask_value_in_pngs"] == 1
    assert segs[0]["provenance"] == "manual"
    assert segs[0]["volume_cc"] == 120.5


def test_segmentations_reject_unknown_structure():
    """An unknown structure name is rejected (must exist in config/masks.py)."""
    with pytest.raises(ValueError):
        build_segmentations([{"structure": "NotARealMask", "mask_value": 1}])


def test_segmentations_reject_invalid_mask_value():
    """A mask value not present in config/masks.py is rejected."""
    with pytest.raises(ValueError):
        build_segmentations([{"structure": "Kidney", "mask_value": 9999}])


# --- Task 33: opt-in ConvertJsonlToV2 step ------------------------------------------------------


def _ctx(tmp: str, schema_version: str) -> PipelineContext:
    """Build a PipelineContext for the conversion step with the given schema_version."""
    pa = PipelineArgs(window_center=50, window_width=350)
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp, schema_version=schema_version),
        dataset=DatasetArgs(dataset_uid="00", dataset_name="kits23", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write_v1_jsonl(ctx_step: ConvertJsonlToV2, records: list) -> str:
    """Write the v1 records to the step's json_path and return it."""
    os.makedirs(os.path.dirname(ctx_step.json_path), exist_ok=True)
    with jsonlines.open(ctx_step.json_path, mode="w") as writer:
        for record in records:
            writer.write(record)
    return ctx_step.json_path


def test_step_noop_in_v1_mode():
    """With schema_version 1.0 (default), the JSONL is left exactly as v1."""
    with tempfile.TemporaryDirectory() as tmp:
        step = ConvertJsonlToV2(_ctx(tmp, SCHEMA_VERSION_V1))
        path = _write_v1_jsonl(step, [_v1_record()])
        step.transform(["x"])
        with jsonlines.open(path) as reader:
            records = list(reader)
        assert "schema_version" not in records[0]
        assert records[0] == _v1_record()


def test_step_emits_v2_when_opted_in():
    """With schema_version 2.0, the JSONL is rewritten as v2 and the window is recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        step = ConvertJsonlToV2(_ctx(tmp, SCHEMA_VERSION_V2))
        path = _write_v1_jsonl(step, [_v1_record()])
        returned = step.transform(["x"])
        assert returned == ["x"]
        with jsonlines.open(path) as reader:
            records = list(reader)
        assert records[0]["schema_version"] == SCHEMA_VERSION_V2
        # the actual intensity transform (window 50/350) is recorded
        png = records[0]["series_level_info"]["png_representation"]
        assert png["png_window_center"] == 50
        # and the emitted v2 still round-trips back to the v1 flat fields
        assert v2_to_v1(records[0]) == _v1_record()
