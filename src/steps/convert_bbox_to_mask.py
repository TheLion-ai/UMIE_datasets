"""Convert object-detection bbox annotations (COCO / YOLO / VOC) into UMIE mask PNGs."""

import glob
import json
import os
import xml.etree.ElementTree as ET
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep

# A normalized box: (source class id/name, x1, y1, x2, y2) in absolute pixel coordinates.
Box = tuple[object, int, int, int, int]


class ConvertBboxToMask(BaseStep):
    """Convert object-detection bbox annotations (COCO / YOLO / VOC) into UMIE mask PNGs.

    Boxes are read from the source format named by ``self.format_config.bbox_format``
    (``coco`` | ``yolo`` | ``voc``), mapped to a target colour via
    ``self.format_config.bbox_class_map`` (``{source_class_id_or_name: target_mask_color}``)
    and rasterized into a mask. Boxes are drawn filled when ``bbox_as_filled`` is set,
    otherwise as a one-pixel outline. Multiple boxes for the same image accumulate into a
    single mask. Output is an 8-bit single-channel PNG whose pixel value is the target colour.
    """

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Convert bbox annotations found under the source path into UMIE mask PNGs.

        Dispatches on ``self.format_config.bbox_format``. COCO reads a single JSON describing
        many images; YOLO reads one ``.txt`` per image (sibling image gives its size); VOC
        reads one XML per image (size taken from the XML). One accumulated mask is written per
        image under the dataset's ``Masks`` folders. ``X`` is returned unchanged.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: The unchanged list of image paths.
        """
        bbox_format = self.format_config.bbox_format
        if not bbox_format:
            return X
        print(f"Converting {bbox_format} bounding boxes to masks...")

        if bbox_format == "coco":
            written = self._convert_coco()
        elif bbox_format == "yolo":
            written = self._convert_yolo()
        elif bbox_format == "voc":
            written = self._convert_voc()
        else:
            raise ValueError(f"Unsupported bbox_format: {bbox_format!r}")

        print(f"Bounding-box conversion complete: {written} mask(s) written.")
        return X

    # --- Source-format parsers (return absolute-pixel boxes) -------------------------------
    @staticmethod
    def _boxes_from_coco(coco: dict) -> dict[int, tuple[int, int, list[Box]]]:
        """Parse a COCO annotation dict into per-image (height, width, boxes).

        Args:
            coco (dict): A parsed COCO JSON with ``images`` and ``annotations``.
        Returns:
            dict[int, tuple[int, int, list[Box]]]: image_id -> (height, width, boxes).
        """
        images: dict[int, tuple[int, int, list[Box]]] = {}
        for image in coco.get("images", []):
            images[int(image["id"])] = (int(image["height"]), int(image["width"]), [])
        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in images:
                continue
            x, y, w, h = annotation["bbox"]
            box: Box = (annotation["category_id"], int(x), int(y), int(x + w), int(y + h))
            images[image_id][2].append(box)
        return images

    @staticmethod
    def _boxes_from_yolo(lines: list[str], height: int, width: int) -> list[Box]:
        """Parse YOLO label lines (``class cx cy w h``, normalized) into absolute boxes.

        Args:
            lines (list[str]): Lines of a YOLO ``.txt`` label file.
            height (int): Pixel height of the referenced image.
            width (int): Pixel width of the referenced image.
        Returns:
            list[Box]: Boxes in absolute pixel coordinates.
        """
        boxes: list[Box] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:5])
            x1 = int(round((cx - w / 2) * width))
            y1 = int(round((cy - h / 2) * height))
            x2 = int(round((cx + w / 2) * width))
            y2 = int(round((cy + h / 2) * height))
            boxes.append((class_id, x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _boxes_from_voc(xml_text: str) -> tuple[int, int, list[Box]]:
        """Parse a Pascal VOC XML string into (height, width, boxes).

        Args:
            xml_text (str): The VOC annotation XML.
        Returns:
            tuple[int, int, list[Box]]: image height, width and the boxes it contains.
        """
        root = ET.fromstring(xml_text)
        size = root.find("size")
        height = int(size.findtext("height", "0")) if size is not None else 0
        width = int(size.findtext("width", "0")) if size is not None else 0
        boxes: list[Box] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x1 = int(round(float(bndbox.findtext("xmin", "0"))))
            y1 = int(round(float(bndbox.findtext("ymin", "0"))))
            x2 = int(round(float(bndbox.findtext("xmax", "0"))))
            y2 = int(round(float(bndbox.findtext("ymax", "0"))))
            boxes.append((name, x1, y1, x2, y2))
        return height, width, boxes

    # --- Rendering -------------------------------------------------------------------------
    def _render_mask(self, boxes: list[Box], height: int, width: int) -> np.ndarray:
        """Rasterize boxes into one accumulated 8-bit single-channel mask.

        Each box's source class is mapped to a target colour through
        ``self.format_config.bbox_class_map`` (boxes with an unmapped class are skipped).
        Boxes are drawn filled when ``bbox_as_filled`` is set, otherwise as a 1px outline.

        Args:
            boxes (list[Box]): Boxes in absolute pixel coordinates.
            height (int): Output mask height.
            width (int): Output mask width.
        Returns:
            np.ndarray: The accumulated mask.
        """
        class_map = self.format_config.bbox_class_map or {}
        as_filled = self.format_config.bbox_as_filled
        mask = np.zeros((height, width), dtype=np.uint8)
        for class_key, x1, y1, x2, y2 in boxes:
            color = self._lookup_color(class_map, class_key)
            if color is None:
                continue
            point1 = (int(x1), int(y1))
            point2 = (int(x2), int(y2))
            thickness = -1 if as_filled else 1
            cv2.rectangle(mask, point1, point2, int(color), thickness)
        return mask

    @staticmethod
    def _lookup_color(class_map: dict, class_key: object) -> Optional[int]:
        """Resolve a source class to its target colour, tolerating int/str key forms.

        Args:
            class_map (dict): ``{source_class_id_or_name: target_mask_color}``.
            class_key (object): The box's source class id or name.
        Returns:
            Optional[int]: The target colour, or None when the class is unmapped.
        """
        if class_key in class_map:
            return int(class_map[class_key])
        # COCO category ids may be configured as strings (or vice versa); try both forms.
        if str(class_key) in class_map:
            return int(class_map[str(class_key)])
        try:
            numeric = int(class_key)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return None
        if numeric in class_map:
            return int(class_map[numeric])
        return None

    # --- Per-format drivers ----------------------------------------------------------------
    def _convert_coco(self) -> int:
        """Convert every COCO JSON under the source path into per-image masks.

        Returns:
            int: The number of masks written.
        """
        written = 0
        for json_path in glob.glob(os.path.join(self.source_path, "**/*.json"), recursive=True):
            with open(json_path) as handle:
                coco = json.load(handle)
            if "images" not in coco or "annotations" not in coco:
                continue
            id_to_name = {int(image["id"]): image["file_name"] for image in coco["images"]}
            for image_id, (height, width, boxes) in self._boxes_from_coco(coco).items():
                mask = self._render_mask(boxes, height, width)
                written += self._write_mask(id_to_name[image_id], mask)
        return written

    def _convert_yolo(self) -> int:
        """Convert every YOLO ``.txt`` (with a sibling image) under the source path.

        Returns:
            int: The number of masks written.
        """
        written = 0
        for txt_path in glob.glob(os.path.join(self.source_path, "**/*.txt"), recursive=True):
            image_path = self._sibling_image(txt_path)
            if image_path is None:
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            with open(txt_path) as handle:
                lines = handle.read().splitlines()
            boxes = self._boxes_from_yolo(lines, height, width)
            mask = self._render_mask(boxes, height, width)
            written += self._write_mask(os.path.basename(image_path), mask)
        return written

    def _convert_voc(self) -> int:
        """Convert every Pascal VOC XML under the source path into per-image masks.

        Returns:
            int: The number of masks written.
        """
        written = 0
        for xml_path in glob.glob(os.path.join(self.source_path, "**/*.xml"), recursive=True):
            with open(xml_path) as handle:
                xml_text = handle.read()
            height, width, boxes = self._boxes_from_voc(xml_text)
            if height == 0 or width == 0:
                continue
            mask = self._render_mask(boxes, height, width)
            root = ET.fromstring(xml_text)
            filename = root.findtext("filename") or os.path.splitext(os.path.basename(xml_path))[0]
            written += self._write_mask(filename, mask)
        return written

    @staticmethod
    def _sibling_image(label_path: str) -> Optional[str]:
        """Find an image file sharing the label file's stem (YOLO convention).

        Args:
            label_path (str): Path to a YOLO ``.txt`` label file.
        Returns:
            Optional[str]: The sibling image path, or None when none exists.
        """
        stem = os.path.splitext(label_path)[0]
        for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            candidate = stem + extension
            if os.path.exists(candidate):
                return candidate
        return None

    def _write_mask(self, image_file_name: str, mask: np.ndarray) -> int:
        """Write a rendered mask under the dataset's Masks folder, returning 1 on success.

        The output basename mirrors the source image file name with a ``.png`` extension so it
        pairs with the converted image. Empty masks are still written so each annotated image
        has a paired mask.

        Args:
            image_file_name (str): Source image file name the boxes belong to.
            mask (np.ndarray): The rendered mask.
        Returns:
            int: 1 when a mask was written, 0 otherwise.
        """
        phase_id = next(iter(self.phases))
        phase_name = self.phases[phase_id]
        base = os.path.splitext(os.path.basename(image_file_name))[0] + ".png"
        masks_dir = os.path.join(self.dataset_root, phase_name, self.mask_folder_name)
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(os.path.join(masks_dir, base), mask)
        return 1
