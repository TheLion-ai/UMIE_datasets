"""Tests for the docs/publication generators: datasheets (Task 43) and subset queries (Task 45)."""

import json
import os
import tempfile

import jsonlines

from utils.datasheet import all_dataset_args, generate_datasheet, write_datasheet
from utils.subset import build_manifest, query_subset, write_manifest


# --- Task 43: datasheet generation --------------------------------------------------------------


def test_datasheet_autofills_from_config():
    """A datasheet pulls uid, modality and licence from config without a JSONL."""
    sheet = generate_datasheet("kits23")
    assert "# 00_kits23" in sheet
    assert "| **UMIE ID** | 00 |" in sheet
    assert "| **Modality** | CT |" in sheet
    # licence comes from config/provenance.py
    assert "CC-BY-NC-SA-4.0" in sheet
    # narrative sections are stubbed for an author
    assert "_(to complete)_" in sheet


def test_datasheet_includes_jsonl_stats():
    """When given a JSONL, the datasheet reports record counts and modality/label breakdowns."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = os.path.join(tmp, "00_kits23.jsonl")
        with jsonlines.open(jsonl, mode="w") as writer:
            writer.write({"modality_name": "CT", "labels": [{"Neoplasm": 1}], "mask_path": "m.png"})
            writer.write({"modality_name": "CT", "labels": [], "mask_path": ""})
        sheet = generate_datasheet("kits23", jsonl_path=jsonl)
        assert "| **No. of Images** | 2 |" in sheet
        assert "| **No. of Annotated Images** | 1 |" in sheet
        assert "| CT | 2 |" in sheet  # modality breakdown
        assert "| Neoplasm | 1 |" in sheet  # label frequency


def test_datasheet_written_to_disk():
    """The datasheet writes to <uid>_<name>.md."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_datasheet("kits23", tmp)
        assert path.endswith("00_kits23.md")
        assert os.path.exists(path)


def test_every_configured_dataset_has_a_name():
    """The dataset registry is non-empty and includes a known dataset."""
    datasets = all_dataset_args()
    assert "kits23" in datasets


# --- Task 45: subset query + manifest -----------------------------------------------------------


def _corpus() -> list[dict]:
    return [
        {
            "umie_path": "00_kits23/CT/Images/a.png",
            "dataset_name": "kits23",
            "modality_name": "CT",
            "labels": [{"ClearCellAdenocarcinoma": 1}],
            "mask_path": "m_a.png",
        },
        {
            "umie_path": "00_kits23/CT/Images/b.png",
            "dataset_name": "kits23",
            "modality_name": "CT",
            "labels": [{"NormalityDecriptor": 1}],
        },
        {
            "umie_path": "11_chest_xray14/Xray/Images/c.png",
            "dataset_name": "chest_xray14",
            "modality_name": "Xray",
            "labels": [{"Pneumonia": 1}],
        },
    ]


def test_query_by_modality():
    """Filtering by modality keeps only matching records."""
    subset = query_subset(_corpus(), modality="Xray")
    assert len(subset) == 1
    assert subset[0]["dataset_name"] == "chest_xray14"


def test_query_label_is_hierarchical():
    """Querying a parent label matches descendant subtypes (Task 36 hierarchy reuse)."""
    # ClearCellAdenocarcinoma is a descendant of Neoplasm -> querying Neoplasm matches it.
    subset = query_subset(_corpus(), label="Neoplasm")
    paths = {r["umie_path"] for r in subset}
    assert "00_kits23/CT/Images/a.png" in paths
    # without descendant expansion, the exact label must be present
    assert query_subset(_corpus(), label="Neoplasm", include_label_descendants=False) == []


def test_query_by_license_uses_provenance():
    """Licence filtering falls back to provenance when records lack an explicit license field."""
    subset = query_subset(_corpus(), license="CC-BY-NC-SA-4.0")  # kits23's licence
    assert {r["dataset_name"] for r in subset} == {"kits23"}


def test_manifest_lists_files_without_copying():
    """The manifest carries the file list + JSONL slice and copies no images."""
    subset = query_subset(_corpus(), modality="CT")
    manifest = build_manifest(subset)
    assert manifest["num_records"] == 2
    assert "00_kits23/CT/Images/a.png" in manifest["files"]
    assert "m_a.png" in manifest["masks"]


def test_manifest_written_to_disk():
    """write_manifest produces valid JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "manifest.json")
        write_manifest(query_subset(_corpus(), modality="CT"), out)
        loaded = json.load(open(out))
        assert loaded["num_records"] == 2
