"""Extract DICOM-SEG / RTSTRUCT segmentations into UMIE mask PNGs aligned to their images."""

import glob
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np
import pydicom

from base.step import BaseStep
from config.masks import all_masks


class ConvertDicomSeg(BaseStep):
    """Extract DICOM-SEG / RTSTRUCT segmentations into UMIE mask PNGs aligned to their images.

    A DICOM-SEG object stores one binary plane per (segment, slice) in a multi-frame pixel
    array; an RTSTRUCT stores polygon contours in patient coordinates. Both are decoded into
    one single-channel mask per referenced image, with each structure painted in the target
    colour from ``self.format_config.segmentation_structure_map``
    (``{source_structure_name: target_mask_color}``), falling back to a matching colour in
    ``config/masks.py`` by RadLex name. ``transform`` dispatches on the file ``Modality``
    (``SEG`` vs ``RTSTRUCT``) over the ``.dcm`` files in ``X`` and writes the masks under the
    dataset's ``Masks`` folders. Output pixel values follow ``config/masks.py`` conventions.
    """

    def transform(
        self,
        X: list,  # source file paths
    ) -> list:
        """Decode DICOM-SEG / RTSTRUCT files in ``X`` into per-slice UMIE mask PNGs.

        Each ``.dcm`` whose ``Modality`` is ``SEG`` or ``RTSTRUCT`` is decoded; the resulting
        per-reference masks are written under the dataset's ``Masks`` folder. ``X`` is
        returned unchanged.

        Args:
            X (list): List of source file paths.
        Returns:
            list: The unchanged list of image paths.
        """
        print("Converting DICOM-SEG / RTSTRUCT to masks...")
        written = 0
        for path in X:
            if not str(path).endswith(".dcm"):
                continue
            try:
                dataset = pydicom.dcmread(path)
            except Exception:
                continue
            modality = getattr(dataset, "Modality", None)
            if modality == "SEG":
                written += self._handle_seg(dataset)
            elif modality == "RTSTRUCT":
                written += self._handle_rtstruct(dataset)
        print(f"Segmentation conversion complete: {written} mask(s) written.")
        return X

    # --- Colour resolution -----------------------------------------------------------------
    def _resolve_color(self, structure_name: Optional[str]) -> Optional[int]:
        """Resolve a source structure name to a target mask colour.

        Looks up ``self.format_config.segmentation_structure_map`` first, then falls back to a
        ``config/masks.py`` mask whose ``radlex_name`` matches the structure name.

        Args:
            structure_name (Optional[str]): The SEG segment / RTSTRUCT ROI structure name.
        Returns:
            Optional[int]: The target colour, or None when the structure cannot be mapped.
        """
        if structure_name is None:
            return None
        mapping = self.format_config.segmentation_structure_map or {}
        if structure_name in mapping:
            return int(mapping[structure_name])
        for mask in all_masks:
            if mask.radlex_name == structure_name:
                return int(mask.color)
        return None

    # --- DICOM-SEG path --------------------------------------------------------------------
    def _handle_seg(self, dataset: pydicom.dataset.Dataset) -> int:
        """Decode a SEG dataset to masks and write one PNG per referenced slice.

        Args:
            dataset (pydicom.dataset.Dataset): The SEG dataset.
        Returns:
            int: The number of masks written.
        """
        masks = self._seg_to_masks(dataset)
        written = 0
        for reference, mask in masks.items():
            written += self._write_mask(str(reference), mask)
        return written

    def _segment_colors(self, dataset: pydicom.dataset.Dataset) -> dict[int, int]:
        """Map each SEG segment number to its target colour via SegmentSequence labels.

        Args:
            dataset (pydicom.dataset.Dataset): The SEG dataset.
        Returns:
            dict[int, int]: ``{segment_number: target_color}`` for resolvable segments.
        """
        colors: dict[int, int] = {}
        for segment in getattr(dataset, "SegmentSequence", []) or []:
            number = int(getattr(segment, "SegmentNumber", 0))
            name = getattr(segment, "SegmentLabel", None) or getattr(segment, "SegmentDescription", None)
            color = self._resolve_color(name)
            if color is not None:
                colors[number] = color
        return colors

    def _seg_to_masks(self, dataset: pydicom.dataset.Dataset) -> dict[object, np.ndarray]:
        """Build one mask per referenced slice from a multi-frame SEG dataset.

        For each frame, the referenced segment number is read from
        ``PerFrameFunctionalGroupsSequence -> SegmentIdentificationSequence`` and the
        referenced image SOP UID from ``DerivationImageSequence``; frames are keyed by that
        SOP UID when present, otherwise by frame index. The segment's target colour is written
        wherever that frame's binary plane equals 1, accumulating across segments per slice.

        Args:
            dataset (pydicom.dataset.Dataset): The SEG dataset.
        Returns:
            dict[object, np.ndarray]: Mapping of referenced SOP UID (or frame index) to mask.
        """
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim == 2:  # a single-frame SEG
            pixels = pixels[np.newaxis, ...]
        rows, cols = pixels.shape[1], pixels.shape[2]
        segment_colors = self._segment_colors(dataset)

        per_frame = list(getattr(dataset, "PerFrameFunctionalGroupsSequence", []) or [])
        masks: dict[object, np.ndarray] = {}
        for index, frame_plane in enumerate(pixels):
            frame_group = per_frame[index] if index < len(per_frame) else None
            segment_number = self._frame_segment_number(frame_group)
            color = segment_colors.get(segment_number)
            if color is None:
                continue
            key = self._frame_reference(frame_group, index)
            mask = masks.setdefault(key, np.zeros((rows, cols), dtype=np.uint8))
            mask[frame_plane.astype(bool)] = color
        return masks

    @staticmethod
    def _frame_segment_number(frame_group: Optional[pydicom.dataset.Dataset]) -> int:
        """Read a frame's referenced segment number from its functional groups.

        Args:
            frame_group (Optional[pydicom.dataset.Dataset]): A PerFrameFunctionalGroups item.
        Returns:
            int: The referenced segment number, or 0 when unavailable.
        """
        if frame_group is None:
            return 0
        seg_id = getattr(frame_group, "SegmentIdentificationSequence", None)
        if not seg_id:
            return 0
        return int(getattr(seg_id[0], "ReferencedSegmentNumber", 0))

    @staticmethod
    def _frame_reference(frame_group: Optional[pydicom.dataset.Dataset], index: int) -> object:
        """Resolve the key identifying the image a SEG frame is aligned to.

        Prefers the referenced source-image SOP UID (from ``DerivationImageSequence ->
        SourceImageSequence``); falls back to the frame index when no reference is present.

        Args:
            frame_group (Optional[pydicom.dataset.Dataset]): A PerFrameFunctionalGroups item.
            index (int): The frame's index in the pixel array.
        Returns:
            object: The referenced SOP Instance UID, or the frame index.
        """
        if frame_group is not None:
            derivation = getattr(frame_group, "DerivationImageSequence", None)
            if derivation:
                source = getattr(derivation[0], "SourceImageSequence", None)
                if source:
                    uid = getattr(source[0], "ReferencedSOPInstanceUID", None)
                    if uid:
                        return str(uid)
        return index

    # --- RTSTRUCT path ---------------------------------------------------------------------
    def _handle_rtstruct(self, dataset: pydicom.dataset.Dataset) -> int:
        """Decode an RTSTRUCT dataset into per-slice masks and write them.

        Each ``ROIContourSequence`` item's contours are rasterized onto the masks of the
        referenced images, looked up from the SOP UIDs stored in the RTSTRUCT's referenced
        image datasets, which must be discoverable next to the RTSTRUCT file.

        Args:
            dataset (pydicom.dataset.Dataset): The RTSTRUCT dataset.
        Returns:
            int: The number of masks written.
        """
        roi_names = self._roi_names(dataset)
        image_geometry = self._reference_image_geometry(dataset)
        masks: dict[str, np.ndarray] = {}
        for roi_contour in getattr(dataset, "ROIContourSequence", []) or []:
            roi_number = int(getattr(roi_contour, "ReferencedROINumber", 0))
            color = self._resolve_color(roi_names.get(roi_number))
            if color is None:
                continue
            for contour in getattr(roi_contour, "ContourSequence", []) or []:
                self._rasterize_into(contour, image_geometry, color, masks)
        written = 0
        for sop_uid, mask in masks.items():
            written += self._write_mask(sop_uid, mask)
        return written

    @staticmethod
    def _roi_names(dataset: pydicom.dataset.Dataset) -> dict[int, str]:
        """Map ROI numbers to their structure names from StructureSetROISequence.

        Args:
            dataset (pydicom.dataset.Dataset): The RTSTRUCT dataset.
        Returns:
            dict[int, str]: ``{roi_number: roi_name}``.
        """
        names: dict[int, str] = {}
        for roi in getattr(dataset, "StructureSetROISequence", []) or []:
            number = int(getattr(roi, "ROINumber", 0))
            names[number] = str(getattr(roi, "ROIName", ""))
        return names

    def _reference_image_geometry(self, dataset: pydicom.dataset.Dataset) -> dict[str, dict]:
        """Collect per-referenced-image geometry needed to rasterize contours.

        Reads the referenced image SOP UIDs from the RTSTRUCT and loads each referenced DICOM
        from the same directory as the RTSTRUCT (matched by SOP Instance UID) to obtain
        ``ImagePositionPatient`` / ``ImageOrientationPatient`` / ``PixelSpacing`` / size.

        Args:
            dataset (pydicom.dataset.Dataset): The RTSTRUCT dataset.
        Returns:
            dict[str, dict]: ``{sop_uid: geometry}`` keyed by referenced image SOP UID.
        """
        wanted = self._referenced_sop_uids(dataset)
        geometry: dict[str, dict] = {}
        directory = os.path.dirname(str(getattr(dataset, "filename", "")) or ".")
        for path in glob.glob(os.path.join(directory, "*.dcm")):
            try:
                image = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                continue
            sop_uid = str(getattr(image, "SOPInstanceUID", ""))
            if sop_uid and (not wanted or sop_uid in wanted) and hasattr(image, "ImagePositionPatient"):
                geometry[sop_uid] = self._geometry_from_image(image)
        return geometry

    @staticmethod
    def _geometry_from_image(image: pydicom.dataset.Dataset) -> dict:
        """Extract the geometry fields used by the patient->pixel mapping from an image.

        Args:
            image (pydicom.dataset.Dataset): A referenced image dataset.
        Returns:
            dict: ``image_position`` / ``image_orientation`` / ``pixel_spacing`` / rows / cols.
        """
        return {
            "image_position": [float(value) for value in image.ImagePositionPatient],
            "image_orientation": [float(value) for value in image.ImageOrientationPatient],
            "pixel_spacing": [float(value) for value in image.PixelSpacing],
            "rows": int(image.Rows),
            "cols": int(image.Columns),
        }

    @staticmethod
    def _referenced_sop_uids(dataset: pydicom.dataset.Dataset) -> set[str]:
        """Gather every referenced image SOP Instance UID from an RTSTRUCT.

        Args:
            dataset (pydicom.dataset.Dataset): The RTSTRUCT dataset.
        Returns:
            set[str]: The referenced SOP Instance UIDs (possibly empty).
        """
        uids: set[str] = set()
        for ref_frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []) or []:
            for study in getattr(ref_frame, "RTReferencedStudySequence", []) or []:
                for series in getattr(study, "RTReferencedSeriesSequence", []) or []:
                    for contour_image in getattr(series, "ContourImageSequence", []) or []:
                        uid = getattr(contour_image, "ReferencedSOPInstanceUID", None)
                        if uid:
                            uids.add(str(uid))
        return uids

    def _rasterize_into(
        self,
        contour: pydicom.dataset.Dataset,
        image_geometry: dict[str, dict],
        color: int,
        masks: dict[str, np.ndarray],
    ) -> None:
        """Rasterize one contour onto the mask of its referenced image.

        Args:
            contour (pydicom.dataset.Dataset): A ContourSequence item.
            image_geometry (dict[str, dict]): ``{sop_uid: geometry}`` for referenced images.
            color (int): The ROI's target colour.
            masks (dict[str, np.ndarray]): Mutable per-SOP-UID mask accumulator.
        """
        sop_uid = self._contour_sop_uid(contour)
        if sop_uid is None or sop_uid not in image_geometry:
            return
        geometry = image_geometry[sop_uid]
        points = [float(value) for value in getattr(contour, "ContourData", [])]
        points_xyz: list[tuple[float, float, float]] = [
            (points[i], points[i + 1], points[i + 2]) for i in range(0, len(points) - 2, 3)
        ]
        polygon = self._rasterize_contour(
            points_xyz,
            geometry["image_position"],
            geometry["image_orientation"],
            geometry["pixel_spacing"],
            geometry["rows"],
            geometry["cols"],
            color,
        )
        mask = masks.setdefault(sop_uid, np.zeros((geometry["rows"], geometry["cols"]), dtype=np.uint8))
        mask[polygon > 0] = color

    @staticmethod
    def _contour_sop_uid(contour: pydicom.dataset.Dataset) -> Optional[str]:
        """Read the referenced image SOP UID a contour lies on.

        Args:
            contour (pydicom.dataset.Dataset): A ContourSequence item.
        Returns:
            Optional[str]: The referenced SOP Instance UID, or None.
        """
        contour_images = getattr(contour, "ContourImageSequence", None)
        if not contour_images:
            return None
        uid = getattr(contour_images[0], "ReferencedSOPInstanceUID", None)
        return str(uid) if uid else None

    @staticmethod
    def _patient_to_pixel(
        point_xyz: tuple[float, float, float],
        image_position: list[float],
        image_orientation: list[float],
        pixel_spacing: list[float],
    ) -> tuple[int, int]:
        """Map a patient-coordinate point to (row, col) pixel indices.

        Uses the standard DICOM image-plane model: the in-plane row/column direction cosines
        from ``ImageOrientationPatient`` project the offset of the point from
        ``ImagePositionPatient`` onto the image grid, scaled by ``PixelSpacing`` (rows, cols).

        Args:
            point_xyz (tuple[float, float, float]): Point in patient coordinates.
            image_position (list[float]): The image's ImagePositionPatient.
            image_orientation (list[float]): The image's ImageOrientationPatient (6 values).
            pixel_spacing (list[float]): The image's PixelSpacing ``[row_spacing, col_spacing]``.
        Returns:
            tuple[int, int]: The (row, col) pixel indices.
        """
        row_cosine = np.array(image_orientation[0:3], dtype=float)
        col_cosine = np.array(image_orientation[3:6], dtype=float)
        origin = np.array(image_position, dtype=float)
        offset = np.array(point_xyz, dtype=float) - origin
        row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
        # DICOM: first ImageOrientationPatient triplet is the across-columns (x) direction,
        # the second is the down-rows (y) direction.
        col = float(np.dot(offset, row_cosine)) / col_spacing
        row = float(np.dot(offset, col_cosine)) / row_spacing
        return int(round(row)), int(round(col))

    @classmethod
    def _rasterize_contour(
        cls,
        points_xyz: list[tuple[float, float, float]],
        image_position: list[float],
        image_orientation: list[float],
        pixel_spacing: list[float],
        rows: int,
        cols: int,
        color: int,
    ) -> np.ndarray:
        """Rasterize a patient-coordinate polygon into a filled mask of the given colour.

        Args:
            points_xyz (list[tuple[float, float, float]]): Polygon vertices in patient coords.
            image_position (list[float]): The image's ImagePositionPatient.
            image_orientation (list[float]): The image's ImageOrientationPatient (6 values).
            pixel_spacing (list[float]): The image's PixelSpacing ``[row_spacing, col_spacing]``.
            rows (int): Output mask height.
            cols (int): Output mask width.
            color (int): The fill colour.
        Returns:
            np.ndarray: The rasterized mask.
        """
        mask = np.zeros((rows, cols), dtype=np.uint8)
        if len(points_xyz) < 3:
            return mask
        pixel_points = [
            cls._patient_to_pixel(point, image_position, image_orientation, pixel_spacing) for point in points_xyz
        ]
        # cv2.fillPoly expects (x, y) == (col, row) vertices.
        polygon = np.array([[col, row] for row, col in pixel_points], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], (int(color),))
        return mask

    # --- Output ----------------------------------------------------------------------------
    def _write_mask(self, reference: str, mask: np.ndarray) -> int:
        """Write a decoded mask under the dataset's Masks folder, returning 1 on success.

        Args:
            reference (str): Referenced image identifier (SOP UID or frame index) used as stem.
            mask (np.ndarray): The decoded mask.
        Returns:
            int: 1 when a mask was written, 0 otherwise.
        """
        modality_id = next(iter(self.modalities))
        modality_name = self.modalities[modality_id]
        masks_dir = os.path.join(self.dataset_root, modality_name, self.mask_folder_name)
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(os.path.join(masks_dir, f"{reference}.png"), mask)
        return 1
