"""Unit tests for MergeMasks using small synthetic single-structure masks (no external data)."""

import json
import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.merge_masks import MergeMasks


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


def test_merge_non_overlapping_combines_colors():
    """Two non-overlapping masks combine into one mask with both colours, overlap_count==0."""
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:3, 0:3] = 1
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_b[6:9, 6:9] = 2

    merged, overlap = MergeMasks._merge([mask_a, mask_b], "report", None)
    assert overlap == 0
    assert merged[1, 1] == 1
    assert merged[7, 7] == 2
    assert set(np.unique(merged).tolist()) == {0, 1, 2}


def test_merge_overlap_first_keeps_earlier():
    """With policy 'first' an overlapping pixel keeps the earlier mask's colour."""
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:5, 0:5] = 1
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_b[3:8, 3:8] = 2

    merged, overlap = MergeMasks._merge([mask_a, mask_b], "first", None)
    assert overlap == 4  # the 2x2 intersection [3:5, 3:5]
    assert merged[4, 4] == 1  # overlap kept the earlier colour
    assert merged[1, 1] == 1
    assert merged[6, 6] == 2


def test_merge_overlap_last_keeps_later():
    """With policy 'last' an overlapping pixel keeps the later mask's colour."""
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:5, 0:5] = 1
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_b[3:8, 3:8] = 2

    merged, overlap = MergeMasks._merge([mask_a, mask_b], "last", None)
    assert overlap == 4
    assert merged[4, 4] == 2  # overlap kept the later colour
    assert merged[1, 1] == 1
    assert merged[6, 6] == 2


def test_merge_overlap_priority_keeps_higher_priority():
    """With policy 'priority' an overlapping pixel keeps the higher-priority colour."""
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_a[0:5, 0:5] = 1
    mask_b = np.zeros((10, 10), dtype=np.uint8)
    mask_b[3:8, 3:8] = 2

    # colour 2 is listed first -> higher priority than colour 1.
    merged, overlap = MergeMasks._merge([mask_a, mask_b], "priority", [2, 1])
    assert overlap == 4
    assert merged[4, 4] == 2  # higher-priority colour wins regardless of order
    assert merged[1, 1] == 1
    assert merged[6, 6] == 2

    # reversing the priority makes colour 1 win the overlap.
    merged2, _ = MergeMasks._merge([mask_a, mask_b], "priority", [1, 2])
    assert merged2[4, 4] == 1


def test_transform_writes_report_and_merged_masks():
    """transform merges grouped masks on disk and writes the overlaps summary report."""
    with tempfile.TemporaryDirectory() as tmp:
        masks_dir = os.path.join(tmp, "99_synthetic", "CT", "Masks")
        os.makedirs(masks_dir, exist_ok=True)

        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_a[0:3, 0:3] = 1
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_b[6:9, 6:9] = 2
        # Same UMIE id under sibling structure subfolders so they group together.
        os.makedirs(os.path.join(masks_dir, "Kidney"), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, "Neoplasm"), exist_ok=True)
        path_a = os.path.join(masks_dir, "Kidney", "99_0_001_img.png")
        path_b = os.path.join(masks_dir, "Neoplasm", "99_0_001_img.png")
        cv2.imwrite(path_a, mask_a)
        cv2.imwrite(path_b, mask_b)

        returned = MergeMasks(_make_ctx(tmp)).transform(["unused"])
        assert returned == ["unused"]

        merged = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(merged).tolist()) == {0, 1, 2}

        report = json.load(open(os.path.join(tmp, "99_synthetic", "reports", "mask_merge_report.json")))
        assert report["overlap_policy"] == "report"
        assert len(report["groups"]) == 1
        assert report["groups"][0]["overlap_pixel_count"] == 0
