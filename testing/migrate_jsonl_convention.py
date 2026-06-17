"""Migrate JSONL goldens between the ``phase_name`` and ``modality_name`` conventions (Task 47/48).

The ``phase`` -> ``modality`` rename (Task 47) changes exactly one emitted JSONL key:
``phase_name`` -> ``modality_name`` (the UMIE id values, folder layout, images and every other field
are unchanged). This script flips that key across every ``.jsonl`` under a directory **in place**,
parsing each record as JSON so only the key is renamed - values, key order and all other fields are
preserved, and a ``phase_name`` appearing inside a *value* is never touched (unlike a raw ``sed``).

It is the repeatable migration step for regenerating the ``umie-tests`` S3 golden data when rolling
the rename out (Task 48): download the bucket, run this over it, re-upload.

Usage:
    python testing/migrate_jsonl_convention.py <dir>             # phase_name -> modality_name
    python testing/migrate_jsonl_convention.py <dir> --reverse   # modality_name -> phase_name
    python testing/migrate_jsonl_convention.py <dir> --dry-run   # report only, write nothing
"""

import argparse
import glob
import os

import jsonlines

PHASE_KEY = "phase_name"
MODALITY_KEY = "modality_name"


def rename_key(record: dict, old: str, new: str) -> tuple[dict, bool]:
    """Return ``record`` with key ``old`` renamed to ``new`` (order preserved) and whether it changed."""
    if old not in record:
        return record, False
    return {new if key == old else key: value for key, value in record.items()}, True


def migrate_file(path: str, old: str, new: str, dry_run: bool = False) -> int:
    """Rename ``old`` -> ``new`` in one JSONL file in place. Returns the number of records changed."""
    with jsonlines.open(path) as reader:
        records = list(reader)
    changed = 0
    migrated = []
    for record in records:
        record, did_change = rename_key(record, old, new)
        changed += int(did_change)
        migrated.append(record)
    if changed and not dry_run:
        with jsonlines.open(path, mode="w") as writer:
            for record in migrated:
                writer.write(record)
    return changed


def migrate_tree(root: str, reverse: bool = False, dry_run: bool = False) -> dict:
    """Migrate every ``.jsonl`` under ``root``. Returns ``{files_changed, records_changed}``."""
    old, new = (MODALITY_KEY, PHASE_KEY) if reverse else (PHASE_KEY, MODALITY_KEY)
    files_changed = 0
    records_changed = 0
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        count = migrate_file(path, old, new, dry_run)
        if count:
            files_changed += 1
            records_changed += count
            print(f"{'[dry-run] ' if dry_run else ''}{path}: {count} records {old} -> {new}")
    return {"files_changed": files_changed, "records_changed": records_changed}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Flip JSONL goldens between phase_name and modality_name (Task 47/48)."
    )
    parser.add_argument("directory", help="Root of the golden .jsonl files (e.g. testing/test_dummy_data)")
    parser.add_argument(
        "--reverse", action="store_true", help="modality_name -> phase_name (default is the forward rename)"
    )
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()

    old, new = (MODALITY_KEY, PHASE_KEY) if args.reverse else (PHASE_KEY, MODALITY_KEY)
    print(f"Migrating *.jsonl under {args.directory!r}: {old} -> {new}")
    result = migrate_tree(args.directory, reverse=args.reverse, dry_run=args.dry_run)
    verb = "would be changed" if args.dry_run else "changed"
    print(f"Done: {result['records_changed']} records in {result['files_changed']} files {verb}.")


if __name__ == "__main__":
    main()
