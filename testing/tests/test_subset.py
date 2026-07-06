"""Tests for the corpus subset/query tool (Theme N, Task 45).

Covers that querying and manifest-building work for both the flat v1 and the hierarchical v2
record schema - in particular that ``build_manifest`` resolves image/mask paths from a v2 record
instead of silently returning empty file lists.
"""

from utils.subset import build_manifest, query_subset


def _v1_record() -> dict:
    return {
        "umie_path": "00_kits23/CT/Images/00_0_001_5.png",
        "mask_path": "00_kits23/CT/Masks/00_0_001_5.png",
        "modality_name": "CT",
        "labels": [{"Neoplasm": 1}],
    }


def _v2_record() -> dict:
    return {
        "schema_version": "2.0",
        "series_level_info": {
            "modality": "CT",
            "png_representation": {
                "type": "native_2d",
                "single_image_path": "00_kits23/CT/Images/00_0_001_5.png",
                "single_mask_path": "00_kits23/CT/Masks/00_0_001_5.png",
            },
        },
        "annotations": {
            "image_level_classification": [{"radlex_label": {"term": "Neoplasm", "id": "RID3957"}, "grade": 1}]
        },
    }


def test_build_manifest_v1():
    """A v1 subset yields the flat umie_path / mask_path."""
    manifest = build_manifest([_v1_record()])
    assert manifest["num_records"] == 1
    assert manifest["files"] == ["00_kits23/CT/Images/00_0_001_5.png"]
    assert manifest["masks"] == ["00_kits23/CT/Masks/00_0_001_5.png"]


def test_build_manifest_v2_resolves_nested_paths():
    """A v2 subset must resolve the nested png/nifti paths, not report num_records>0 with empty files."""
    manifest = build_manifest([_v2_record()])
    assert manifest["num_records"] == 1
    assert manifest["files"] == ["00_kits23/CT/Images/00_0_001_5.png"]
    assert manifest["masks"] == ["00_kits23/CT/Masks/00_0_001_5.png"]


def test_query_then_manifest_v2_end_to_end():
    """query_subset (v2-aware) + build_manifest produce a non-empty manifest for a matching v2 record."""
    subset = query_subset([_v2_record()], modality="CT", label="Neoplasm")
    manifest = build_manifest(subset)
    assert manifest["num_records"] == 1
    assert manifest["files"] and manifest["masks"]
