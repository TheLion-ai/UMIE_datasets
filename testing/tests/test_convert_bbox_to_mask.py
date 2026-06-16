"""Unit tests for ConvertBboxToMask across COCO / YOLO / VOC (no external data)."""

import os
import tempfile

import cv2
import numpy as np

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.convert_bbox_to_mask import ConvertBboxToMask


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


def test_render_mask_filled_box_at_correct_coords_and_color():
    """A single mapped box renders a filled rectangle of the mapped colour at its coords."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.bbox_class_map = {7: 5}
        ctx.format_conversion.bbox_as_filled = True
        step = ConvertBboxToMask(ctx)

        mask = step._render_mask([(7, 2, 3, 6, 8)], height=20, width=20)
        assert mask[3, 2] == 5  # top-left corner inside the box
        assert mask[8, 6] == 5  # bottom-right corner inside the box
        assert mask[5, 4] == 5  # interior is filled
        assert mask[0, 0] == 0  # outside the box stays background


def test_coco_multi_box_accumulation():
    """Multiple COCO boxes for one image accumulate into a single mask."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.bbox_class_map = {1: 5, 2: 7}
        step = ConvertBboxToMask(ctx)

        coco = {
            "images": [{"id": 10, "file_name": "scan.png", "height": 30, "width": 30}],
            "annotations": [
                {"image_id": 10, "category_id": 1, "bbox": [2, 2, 5, 5]},
                {"image_id": 10, "category_id": 2, "bbox": [20, 20, 5, 5]},
            ],
        }
        parsed = step._boxes_from_coco(coco)
        height, width, boxes = parsed[10]
        assert (height, width) == (30, 30)
        mask = step._render_mask(boxes, height, width)
        assert mask[4, 4] == 5  # first box colour
        assert mask[22, 22] == 7  # second box colour
        assert mask[15, 15] == 0  # gap between boxes


def test_yolo_normalized_box_maps_to_pixels():
    """A YOLO normalized box maps to the expected absolute pixel rectangle and colour."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.bbox_class_map = {0: 5}
        step = ConvertBboxToMask(ctx)

        # class 0, centre (0.5, 0.5), size 0.5x0.5 on a 100x100 image -> [25,25]..[75,75].
        boxes = step._boxes_from_yolo(["0 0.5 0.5 0.5 0.5"], height=100, width=100)
        assert boxes == [(0, 25, 25, 75, 75)]
        mask = step._render_mask(boxes, 100, 100)
        assert mask[50, 50] == 5
        assert mask[10, 10] == 0


def test_voc_xml_box_parsed_and_rendered():
    """A Pascal VOC XML object yields the right size, box and rendered colour."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.bbox_class_map = {"lesion": 11}
        step = ConvertBboxToMask(ctx)

        xml_text = (
            "<annotation><filename>scan.png</filename>"
            "<size><height>40</height><width>40</width></size>"
            "<object><name>lesion</name>"
            "<bndbox><xmin>5</xmin><ymin>6</ymin><xmax>15</xmax><ymax>16</ymax></bndbox>"
            "</object></annotation>"
        )
        height, width, boxes = step._boxes_from_voc(xml_text)
        assert (height, width) == (40, 40)
        assert boxes == [("lesion", 5, 6, 15, 16)]
        mask = step._render_mask(boxes, height, width)
        assert mask[10, 10] == 11
        assert mask[0, 0] == 0


def test_transform_coco_writes_mask_png():
    """End-to-end COCO transform writes an accumulated mask PNG under the Masks folder."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        ctx.format_conversion.bbox_format = "coco"
        ctx.format_conversion.bbox_class_map = {1: 5}

        coco = {
            "images": [{"id": 1, "file_name": "scan.png", "height": 20, "width": 20}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [4, 4, 6, 6]}],
        }
        import json as _json

        with open(os.path.join(tmp, "annotations.json"), "w") as handle:
            _json.dump(coco, handle)

        returned = ConvertBboxToMask(ctx).transform(["unused"])
        assert returned == ["unused"]

        mask_path = os.path.join(tmp, "99_synthetic", "CT", "Masks", "scan.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert mask[6, 6] == 5
        assert int(np.count_nonzero(mask)) > 0
