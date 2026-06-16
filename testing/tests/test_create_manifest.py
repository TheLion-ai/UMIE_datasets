"""Unit tests for CreateManifest: hashing, reports/ exclusion, and verify-mode categories."""

import hashlib
import json
import os
import tempfile

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.create_manifest import CreateManifest


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _write(path: str, data: bytes) -> None:
    """Create parent dirs and write ``data`` to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def _sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def test_manifest_hashes_match_and_reports_excluded():
    """Manifest digests equal a hashlib recompute; reports/ and the manifest are excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        step = CreateManifest(ctx)
        root = step.dataset_root

        img = os.path.join(root, "CT", "Images", "99_0_001_a.png")
        jsonl = os.path.join(root, "99_synthetic.jsonl")
        _write(img, b"image-bytes")
        _write(jsonl, b'{"umie_id": "99_0_001_a.png"}\n')

        # A pre-existing report file must NOT appear in the manifest.
        _write(os.path.join(step.reports_dir(), "some_report.json"), b"{}")

        returned = step.transform(["unchanged"])
        assert returned == ["unchanged"]  # X passes through untouched

        manifest_path = os.path.join(step.reports_dir(), "manifest.json")
        manifest = json.load(open(manifest_path))

        assert manifest["CT/Images/99_0_001_a.png"] == _sha256(b"image-bytes")
        assert manifest["99_synthetic.jsonl"] == _sha256(b'{"umie_id": "99_0_001_a.png"}\n')

        # Nothing under reports/ (incl. the manifest itself) is in the manifest.
        assert not any(rel.startswith("reports/") for rel in manifest)
        assert "reports/manifest.json" not in manifest
        assert "reports/some_report.json" not in manifest


def test_verify_reports_missing_changed_and_extra():
    """Verify mode flags a changed, a missing, and an extra file correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        step = CreateManifest(ctx)
        root = step.dataset_root

        keep = os.path.join(root, "CT", "Images", "keep.png")
        to_change = os.path.join(root, "CT", "Images", "change.png")
        to_delete = os.path.join(root, "CT", "Images", "delete.png")
        _write(keep, b"keep")
        _write(to_change, b"original")
        _write(to_delete, b"delete-me")

        # Build the baseline manifest.
        step.transform([])

        # Mutate the outputs: change one, delete one, add one.
        _write(to_change, b"MODIFIED")
        os.remove(to_delete)
        added = os.path.join(root, "CT", "Images", "extra.png")
        _write(added, b"new-file")

        # Switch to verify mode (manifest must not be rewritten).
        manifest_path = os.path.join(step.reports_dir(), "manifest.json")
        before = open(manifest_path).read()
        ctx.export.verify_manifest = True
        CreateManifest(ctx).transform([])
        assert open(manifest_path).read() == before  # manifest untouched in verify mode

        report = json.load(open(os.path.join(step.reports_dir(), "manifest_verification.json")))
        assert report["changed"] == ["CT/Images/change.png"]
        assert report["missing"] == ["CT/Images/delete.png"]
        assert report["extra"] == ["CT/Images/extra.png"]
