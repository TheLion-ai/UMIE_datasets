"""Tests for the phase_name <-> modality_name JSONL migration tool (Task 47/48)."""

import os
import tempfile

import jsonlines

from testing.migrate_jsonl_convention import migrate_file, migrate_tree, rename_key


def _write(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record)


def _read(path: str) -> list:
    with jsonlines.open(path) as reader:
        return list(reader)


def test_rename_key_preserves_order_and_values():
    """Only the key is renamed; position and all values are preserved."""
    record = {"umie_path": "x", "phase_name": "CT", "study_id": "1"}
    out, changed = rename_key(record, "phase_name", "modality_name")
    assert changed
    assert list(out.keys()) == ["umie_path", "modality_name", "study_id"]
    assert out["modality_name"] == "CT"
    # a record without the key is untouched
    assert rename_key({"a": 1}, "phase_name", "modality_name") == ({"a": 1}, False)


def test_value_named_phase_name_is_not_touched():
    """A 'phase_name' string appearing in a value must not be altered (unlike a raw sed)."""
    record = {"phase_name": "CT", "note": "phase_name in a value"}
    out, _ = rename_key(record, "phase_name", "modality_name")
    assert out["note"] == "phase_name in a value"
    assert "modality_name" in out and "phase_name" not in out


def test_migrate_file_forward():
    """Forward migration renames phase_name -> modality_name in every record."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "00_ds.jsonl")
        _write(path, [{"phase_name": "CT", "labels": []}, {"phase_name": "MRI", "labels": []}])
        assert migrate_file(path, "phase_name", "modality_name") == 2
        out = _read(path)
        assert all("modality_name" in r and "phase_name" not in r for r in out)
        assert [r["modality_name"] for r in out] == ["CT", "MRI"]


def test_migrate_tree_and_reverse_round_trip():
    """Migrating a tree forward then reverse restores the original records exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "07_lits.jsonl")
        original = [{"umie_path": "a", "phase_name": "CT", "study_id": "1"}]
        _write(path, original)

        forward = migrate_tree(tmp)
        assert forward == {"files_changed": 1, "records_changed": 1}
        assert _read(path)[0] == {"umie_path": "a", "modality_name": "CT", "study_id": "1"}

        reverse = migrate_tree(tmp, reverse=True)
        assert reverse == {"files_changed": 1, "records_changed": 1}
        assert _read(path) == original


def test_dry_run_writes_nothing():
    """--dry-run reports the change count without modifying any file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.jsonl")
        _write(path, [{"phase_name": "CT"}])
        result = migrate_tree(tmp, dry_run=True)
        assert result == {"files_changed": 1, "records_changed": 1}
        assert _read(path) == [{"phase_name": "CT"}]  # unchanged on disk
