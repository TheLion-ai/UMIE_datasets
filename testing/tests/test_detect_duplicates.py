"""Unit tests for DetectDuplicates using small synthetic PNGs (no external data)."""

import json
import os
import tempfile

import cv2
import jsonlines
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.detect_duplicates import DetectDuplicates


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", modalities={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _images_dir(tmp: str) -> str:
    """Create and return the synthetic Images folder."""
    path = os.path.join(tmp, "99_synthetic", "CT", "Images")
    os.makedirs(path, exist_ok=True)
    return path


def _gradient(seed: int) -> np.ndarray:
    """Build a deterministic 64x64 grayscale gradient image distinct per seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


def _write_synthetic(images_dir: str) -> dict:
    """Write an exact-duplicate pair, a near-duplicate pair and a distinct image."""
    base = _gradient(1)
    near = base.copy()
    # invert a small corner block -> a near duplicate at dHash distance 3 (within the
    # default threshold of 5, but excluded at threshold 0).
    near[:16, :16] = 255 - near[:16, :16]
    distinct = _gradient(99)

    paths = {
        "exact_a": os.path.join(images_dir, "99_0_001_a.png"),
        "exact_b": os.path.join(images_dir, "99_0_002_b.png"),
        "near": os.path.join(images_dir, "99_0_003_c.png"),
        "distinct": os.path.join(images_dir, "99_0_004_d.png"),
    }
    cv2.imwrite(paths["exact_a"], base)
    cv2.imwrite(paths["exact_b"], base)
    cv2.imwrite(paths["near"], near)
    cv2.imwrite(paths["distinct"], distinct)
    return paths


def _rel(path: str, tmp: str) -> str:
    """Return the umie_path (relative to the target/tmp dir) of an output file."""
    return os.path.relpath(path, tmp).replace(os.sep, "/")


def test_duplicates_cluster_and_distinct_does_not():
    """Exact and near duplicates cluster together; the distinct image stays out."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        paths = _write_synthetic(images_dir)
        ctx = _make_ctx(tmp)

        returned = DetectDuplicates(ctx).transform(list(paths.values()))
        assert returned == list(paths.values())  # X returned unchanged

        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "duplicates_report.json")))
        clusters = report["clusters"]
        assert len(clusters) == 1
        members = set(clusters[0]["members"])
        assert _rel(paths["exact_a"], tmp) in members
        assert _rel(paths["exact_b"], tmp) in members
        assert _rel(paths["near"], tmp) in members
        assert _rel(paths["distinct"], tmp) not in members


def test_nothing_deleted_and_csv_written():
    """Detection never deletes any image and writes both JSON and CSV reports."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        paths = _write_synthetic(images_dir)
        DetectDuplicates(_make_ctx(tmp)).transform(list(paths.values()))

        for path in paths.values():
            assert os.path.exists(path)
        reports_dir = os.path.join(tmp, "99_synthetic", "reports")
        assert os.path.exists(os.path.join(reports_dir, "duplicates_report.json"))
        assert os.path.exists(os.path.join(reports_dir, "duplicates_report.csv"))


def test_threshold_is_honored():
    """A threshold of 0 only clusters exact duplicates, not the near-duplicate."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        paths = _write_synthetic(images_dir)
        ctx = _make_ctx(tmp)
        ctx.quality.duplicate_threshold = 0

        DetectDuplicates(ctx).transform(list(paths.values()))
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "duplicates_report.json")))
        clusters = report["clusters"]
        assert len(clusters) == 1
        members = set(clusters[0]["members"])
        # exact pair clusters, near-duplicate does not at distance 0
        assert _rel(paths["exact_a"], tmp) in members
        assert _rel(paths["exact_b"], tmp) in members
        assert _rel(paths["near"], tmp) not in members
        assert clusters[0]["max_distance"] == 0


def _seed_jsonl(tmp: str, paths: dict) -> str:
    """Write a synthetic JSONL with one record per image and return its path."""
    json_path = os.path.join(tmp, "99_synthetic", "99_synthetic.jsonl")
    records = [{"umie_path": _rel(path, tmp), "labels": []} for path in paths.values()]
    with jsonlines.open(json_path, mode="w") as writer:
        for record in records:
            writer.write(record)
    return json_path


def test_flag_adds_group_id_only_when_enabled():
    """duplicate_group_id is added to clustered records only when flagging is enabled."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        paths = _write_synthetic(images_dir)
        json_path = _seed_jsonl(tmp, paths)

        # disabled (default): no group id written anywhere
        DetectDuplicates(_make_ctx(tmp)).transform(list(paths.values()))
        with jsonlines.open(json_path, mode="r") as reader:
            assert all("duplicate_group_id" not in obj for obj in reader)

        # enabled: clustered records get a group id, the distinct record does not
        ctx = _make_ctx(tmp)
        ctx.quality.flag_duplicates_in_jsonl = True
        DetectDuplicates(ctx).transform(list(paths.values()))
        with jsonlines.open(json_path, mode="r") as reader:
            by_path = {obj["umie_path"]: obj for obj in reader}
        assert "duplicate_group_id" in by_path[_rel(paths["exact_a"], tmp)]
        assert "duplicate_group_id" in by_path[_rel(paths["near"], tmp)]
        assert "duplicate_group_id" not in by_path[_rel(paths["distinct"], tmp)]
        # all original fields preserved
        assert by_path[_rel(paths["distinct"], tmp)]["labels"] == []


def test_cross_dataset_reference_hashes():
    """An external reference-hash JSON lets cross-dataset overlaps surface in a cluster."""
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = _images_dir(tmp)
        paths = _write_synthetic(images_dir)
        ctx = _make_ctx(tmp)

        step = DetectDuplicates(ctx)
        base_image = cv2.imread(paths["exact_a"], cv2.IMREAD_GRAYSCALE)
        reference = {"other/00_0_001_x.png": step._dhash(base_image)}
        ref_path = os.path.join(tmp, "reference.json")
        json.dump(reference, open(ref_path, "w"))
        ctx.quality.duplicate_reference_hashes = ref_path

        DetectDuplicates(ctx).transform(list(paths.values()))
        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "duplicates_report.json")))
        all_members = {member for cluster in report["clusters"] for member in cluster["members"]}
        assert "other/00_0_001_x.png" in all_members
