"""Versioned hierarchical JSONL metadata schema (Theme K: Tasks 33-35).

The pipeline emits a **flat v1** JSONL record per image (``umie_path``, ``umie_id``,
``modality_name``, ``labels``, ``source_labels``, ...). This module adds an **opt-in v2** record
that mirrors the DICOM patient -> study -> series hierarchy from ``JSONlines.md`` while *reusing*
the data the pipeline already has - it never renames or drops a v1 field, and v1 stays the default.

It provides:

* **Task 33** - :func:`build_v2_record` (v1 flat record -> v2 hierarchical record) and its inverse
  :func:`v2_to_v1`, so a v2 record can always regenerate the v1 flat fields (round-trip).
* **Task 34** - a controlled, per-modality :data:`MODALITY_PROTOCOL_FIELDS` vocabulary and
  :func:`build_imaging_protocol_details`, which emits modality-specific protocol fields with dual
  ``_radlex`` / ``_source`` coding for key terms and validates values against the vocab.
* **Task 35** - :func:`build_png_representation`, :func:`build_nifti_files` (reuses the Task 8
  ``volume_metadata``, no recomputation) and :func:`build_segmentations` with per-segmentation
  dual labels, mask-value mapping and provenance.

The functions are pure (dict in, dict out) so they are independently testable and import nothing
from the pipeline machinery.
"""

from __future__ import annotations

from typing import Any, Optional

from config import labels as labels_config
from config import masks as masks_config
from config.provenance import get_provenance
from constants import SCHEMA_VERSION_V2

# --- Task 34: controlled modality + protocol vocabulary (single source of truth) ----------------

#: Protocol fields that are meaningful for each modality (per ``Modalities Ontology.md``). A field
#: absent from the source is omitted from the record, never faked.
MODALITY_PROTOCOL_FIELDS: dict[str, list[str]] = {
    "MRI": [
        "mri_sequence_type",
        "mri_contrast_agent",
        "mri_contrast_phase",
        "mri_fat_saturation",
        "mri_acquisition_plane",
    ],
    "CT": ["ct_contrast_agent", "ct_contrast_phase", "ct_kernel", "ct_kvp", "ct_low_dose"],
    "Xray": ["xray_projection"],
    "MG": ["mammography_view"],
    "PET": ["pet_radiotracer", "pet_scan_type"],
}

#: Common acquisition fields allowed for every modality.
COMMON_PROTOCOL_FIELDS: list[str] = ["original_slice_thickness_mm", "original_pixel_spacing_mm"]

#: Key categorical terms that carry the dual ``{field}_radlex`` {term,id} + ``{field}_source`` coding.
KEY_PROTOCOL_TERMS: set[str] = {
    "mri_sequence_type",
    "ct_kernel",
    "xray_projection",
    "mammography_view",
    "pet_radiotracer",
}

#: Allowed values for the controlled categorical fields. Values are validated against these.
CONTROLLED_VOCAB: dict[str, set[str]] = {
    "xray_projection": {"AP", "PA", "Lateral", "Oblique"},
    "mammography_view": {"CC", "MLO"},
    "ct_contrast_phase": {"Non-contrast", "Arterial", "Venous", "Delayed"},
    "mri_contrast_phase": {"Pre-contrast", "Post-contrast", "Arterial", "Venous", "Delayed"},
    "mri_acquisition_plane": {"Axial", "Sagittal", "Coronal", "Oblique"},
    "pet_scan_type": {"Static", "Dynamic", "Whole Body"},
}

#: Optional RadLex coding for selected key-term values (``field -> source_value -> {term, id}``).
PROTOCOL_RADLEX: dict[str, dict[str, dict[str, str]]] = {
    "mri_sequence_type": {
        "T1W": {"term": "T1-weighted SE MR imaging", "id": "RID3897"},
        "T2W": {"term": "T2 weighted image", "id": "RID11086"},
        "FLAIR": {"term": "fluid-attenuated inversion recovery MR imaging", "id": "RID11084"},
    },
    "xray_projection": {
        "PA": {"term": "posteroanterior projection", "id": "RID10567"},
        "AP": {"term": "anteroposterior projection", "id": "RID10559"},
    },
}


def build_imaging_protocol_details(modality: str, source_protocol: dict[str, Any], validate: bool = True) -> dict:
    """Build the modality-specific ``imaging_protocol_details`` sub-block (Task 34).

    Args:
        modality: Modality name (``CT`` / ``MRI`` / ``Xray`` / ``MG`` / ``PET``).
        source_protocol: Raw protocol values pulled from the source (DICOM / NIfTI / dataset).
            Keys outside the modality's allowed field set are ignored; ``None`` values are omitted.
        validate: When True, values of controlled categorical fields must be in :data:`CONTROLLED_VOCAB`.

    Returns:
        dict: The protocol block. Key terms carry dual ``{field}_radlex`` / ``{field}_source`` values.

    Raises:
        ValueError: If ``validate`` and a controlled field carries a value outside its vocabulary.
    """
    allowed = set(MODALITY_PROTOCOL_FIELDS.get(modality, [])) | set(COMMON_PROTOCOL_FIELDS)
    details: dict[str, Any] = {}
    for field, value in source_protocol.items():
        if field not in allowed or value is None:
            continue
        if validate and field in CONTROLLED_VOCAB and value not in CONTROLLED_VOCAB[field]:
            raise ValueError(
                f"Value {value!r} for {field} is not in the controlled vocabulary {sorted(CONTROLLED_VOCAB[field])}"
            )
        if field in KEY_PROTOCOL_TERMS:
            details[f"{field}_source"] = value
            radlex = PROTOCOL_RADLEX.get(field, {}).get(value)
            if radlex:
                details[f"{field}_radlex"] = radlex
        else:
            details[field] = value
    return details


# --- Task 35: png_representation / nifti_files / segmentations ----------------------------------


def describe_normalization(
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    bit_depth: int = 8,
) -> str:
    """Return a human-readable description of the intensity transform applied to the PNGs (Task 35)."""
    if window_center is not None and window_width is not None:
        return f"Windowed (WC:{window_center}, WW:{window_width}), then scaled to {bit_depth}-bit"
    return f"Original intensities scaled to {bit_depth}-bit"


def build_png_representation(
    *,
    type: str,
    single_image_path: Optional[str] = None,
    single_mask_path: Optional[str] = None,
    base_image_path_pattern: Optional[str] = None,
    base_mask_path_pattern: Optional[str] = None,
    num_slices: Optional[int] = None,
    slicing_axis: Optional[int] = None,
    png_pixel_spacing_mm: Optional[list] = None,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    rescale_slope: Optional[float] = None,
    png_bit_depth: int = 8,
    normalization_applied_to_png: Optional[str] = None,
) -> dict:
    """Build the ``png_representation`` block describing how PNGs relate to source geometry (Task 35).

    ``type`` is ``"native_2d"`` (e.g. an X-ray) or ``"3d_sliced_to_2d"`` (slices of a volume).
    ``normalization_applied_to_png`` records the *actual* intensity transform; if not supplied it is
    derived from the window/​bit-depth via :func:`describe_normalization`. Absent fields are omitted.
    """
    if type not in ("native_2d", "3d_sliced_to_2d"):
        raise ValueError(f"png_representation type must be 'native_2d' or '3d_sliced_to_2d', got {type!r}")
    block: dict[str, Any] = {"type": type}
    optional = {
        "single_image_path": single_image_path,
        "single_mask_path": single_mask_path,
        "base_image_path_pattern": base_image_path_pattern,
        "base_mask_path_pattern": base_mask_path_pattern,
        "num_slices": num_slices,
        "slicing_axis_of_original_volume": slicing_axis,
        "png_pixel_spacing_mm": png_pixel_spacing_mm,
        "png_window_center": window_center,
        "png_window_width": window_width,
        "png_rescale_intercept": rescale_intercept,
        "png_rescale_slope": rescale_slope,
    }
    block.update({k: v for k, v in optional.items() if v is not None})
    block["png_bit_depth"] = png_bit_depth
    block["normalization_applied_to_png"] = normalization_applied_to_png or describe_normalization(
        window_center, window_width, png_bit_depth
    )
    return block


def build_nifti_files(volume_metadata: dict, image_path: str, mask_path: Optional[str] = None) -> dict:
    """Build the ``nifti_files`` block from the Task 8 ``volume_metadata`` (Task 35, no recomputation).

    Args:
        volume_metadata: The ``{shape, voxel_spacing_mm, orientation, affine}`` dict written by
            ``AddUmieIds._volume_metadata`` in 3D mode (the single source of truth for 3D geometry).
        image_path: Relative path to the NIfTI image volume.
        mask_path: Relative path to the NIfTI mask volume, if any.

    Returns:
        dict: ``nifti_files`` block with affine, dimensions and voxel spacing reused from the header.
    """
    block: dict[str, Any] = {"image_path": image_path}
    if mask_path:
        block["mask_path"] = mask_path
    if "affine" in volume_metadata:
        block["affine_matrix"] = volume_metadata["affine"]
    if "shape" in volume_metadata:
        block["original_nifti_dimensions"] = volume_metadata["shape"]
    if "voxel_spacing_mm" in volume_metadata:
        block["original_nifti_voxel_spacing_mm"] = volume_metadata["voxel_spacing_mm"]
    return block


def build_segmentations(segmentations: list[dict]) -> list[dict]:
    """Build the ``annotations.segmentations`` list with dual labels, mask values and provenance (Task 35).

    Each input dict supplies ``structure`` (RadLex mask name), optional ``structure_source``,
    ``mask_value`` (the value in both NIfTI and PNG masks unless given separately), ``provenance``
    (``manual`` / ``semi_automatic`` / ``model`` ...) and optional ``volume_cc``. ``structure`` and
    ``mask_value`` are validated against ``config/masks.py``.

    Raises:
        ValueError: If a structure name or mask value is not present in ``config/masks.py``.
    """
    valid_names = {m.radlex_name for m in masks_config.all_masks}
    valid_values = {m.id for m in masks_config.all_masks}
    out: list[dict] = []
    for seg in segmentations:
        structure = seg["structure"]
        if structure not in valid_names:
            raise ValueError(f"Segmentation structure {structure!r} is not in config/masks.py")
        mask = masks_config.mask_by_name(structure)
        mask_value_nifti = seg.get("mask_value_in_nifti", seg.get("mask_value", mask.id if mask else None))
        mask_value_pngs = seg.get("mask_value_in_pngs", seg.get("mask_value", mask.id if mask else None))
        for value in (mask_value_nifti, mask_value_pngs):
            if value is not None and value not in valid_values:
                raise ValueError(f"Mask value {value} for {structure!r} is not a valid value in config/masks.py")
        entry: dict[str, Any] = {
            "structure_radlex": {"term": structure, "id": mask.radlex_id if mask else ""},
            "structure_source": seg.get("structure_source", ""),
            "mask_value_in_nifti": mask_value_nifti,
            "mask_value_in_pngs": mask_value_pngs,
            "provenance": seg.get("provenance", "unknown"),
        }
        if seg.get("volume_cc") is not None:
            entry["volume_cc"] = seg["volume_cc"]
        out.append(entry)
    return out


# --- Task 33: v1 <-> v2 record serializer -------------------------------------------------------


def _image_level_classification(v1_labels: list) -> list[dict]:
    """Map v1 ``labels`` (list of ``{radlex_name: grade}``) to ``image_level_classification`` entries."""
    classification: list[dict] = []
    for label_dict in v1_labels:
        for term, grade in label_dict.items():
            label = labels_config.label_by_name(term)
            classification.append(
                {
                    "radlex_label": {"term": term, "id": label.radlex_id if label else ""},
                    "grade": grade,
                }
            )
    return classification


def build_v2_record(
    v1_record: dict,
    *,
    body_part: Optional[dict] = None,
    imaging_protocol_details: Optional[dict] = None,
    png_representation: Optional[dict] = None,
    nifti_files: Optional[dict] = None,
    segmentations: Optional[list[dict]] = None,
    patient_info: Optional[dict] = None,
    study_info: Optional[dict] = None,
    original_filenames: Optional[list[str]] = None,
) -> dict:
    """Build a v2 hierarchical record from a flat v1 record, reusing data the pipeline already has.

    Only ``v1_record`` is required; every other block is optional and supplied by callers that have
    the extra information (protocol from DICOM, geometry from NIfTI, segmentation provenance, ...).
    The UMIE id components (``dataset_uid`` / ``modality`` / ``study`` / ``img``) are **reused** as
    the hierarchy keys, never renamed.

    Args:
        v1_record: A flat v1 JSONL record (as written by ``AddUmieIds`` / ``AddLabels``).

    Returns:
        dict: The v2 record (``schema_version == "2.0"``).
    """
    provenance = get_provenance(v1_record.get("dataset_name", ""))
    study_id = str(v1_record.get("study_id", ""))
    umie_id = v1_record.get("umie_id", "")
    umie_path = v1_record.get("umie_path", "")
    mask_path = v1_record.get("mask_path", "")

    # png_representation / nifti_files default to a sensible native-2D / volume layout when the
    # caller did not pass richer geometry, so the v1 paths are always recoverable on round-trip.
    is_volume = umie_path.endswith(".nii.gz") or bool(v1_record.get("volume_metadata"))
    if png_representation is None and not is_volume:
        png_representation = build_png_representation(
            type="native_2d",
            single_image_path=umie_path,
            single_mask_path=mask_path or None,
        )
    if nifti_files is None and is_volume:
        nifti_files = build_nifti_files(
            v1_record.get("volume_metadata", {}), image_path=umie_path, mask_path=mask_path or None
        )

    series_level_info: dict[str, Any] = {
        "series_id": umie_id,
        "modality": v1_record.get("modality_name", ""),
    }
    if body_part is not None:
        series_level_info["body_part_examined_radlex"] = body_part.get("radlex")
        series_level_info["body_part_examined_source"] = body_part.get("source")
    if imaging_protocol_details:
        series_level_info["imaging_protocol_details"] = imaging_protocol_details
    if nifti_files:
        series_level_info["nifti_files"] = nifti_files
    if png_representation:
        series_level_info["png_representation"] = png_representation

    annotations: dict[str, Any] = {
        "image_level_classification": _image_level_classification(v1_record.get("labels", [])),
        # ``source_labels`` is kept verbatim alongside the classification so the v1 list (which may
        # not be 1:1 with ``labels``) round-trips exactly.
        "source_labels": list(v1_record.get("source_labels", [])),
        "segmentations": build_segmentations(segmentations) if segmentations else [],
    }

    return {
        "schema_version": SCHEMA_VERSION_V2,
        "entry_id": umie_id,
        "dataset_source_details": {
            "name": provenance.source_dataset,
            "dataset_uid": v1_record.get("dataset_uid", ""),
            "dataset_name": v1_record.get("dataset_name", ""),
            "original_filenames": original_filenames or [],
            "original_id_within_source_dataset": (study_info or {}).get("original_id", study_id),
        },
        "patient_level_info": patient_info or {"patient_id": f"{v1_record.get('dataset_uid', '')}_{study_id}"},
        "study_level_info": {
            "study_id": study_id,
            "study_date_offset_days": (study_info or {}).get("study_date_offset_days"),
            "study_description": (study_info or {}).get("study_description"),
            # ``comparative`` (PRE/POST) is preserved so it round-trips without re-deriving from the id.
            "comparative": v1_record.get("comparative", ""),
        },
        "series_level_info": series_level_info,
        "annotations": annotations,
    }


def v2_to_v1(v2_record: dict) -> dict:
    """Regenerate the flat v1 record from a v2 record (Task 33 round-trip).

    Recovers ``umie_path`` / ``mask_path`` from whichever representation block carries them,
    ``labels`` from ``image_level_classification`` and ``source_labels`` verbatim - so a record can
    cross the v1<->v2 boundary losslessly for the v1 fields.
    """
    series = v2_record.get("series_level_info", {})
    png = series.get("png_representation", {})
    nifti = series.get("nifti_files", {})
    umie_path = png.get("single_image_path") or nifti.get("image_path") or ""
    mask_path = png.get("single_mask_path") or nifti.get("mask_path") or ""

    annotations = v2_record.get("annotations", {})
    labels = [{c["radlex_label"]["term"]: c["grade"]} for c in annotations.get("image_level_classification", [])]

    source = v2_record.get("dataset_source_details", {})
    study = v2_record.get("study_level_info", {})
    return {
        "umie_path": umie_path,
        "dataset_name": source.get("dataset_name", source.get("name", "")),
        "dataset_uid": source.get("dataset_uid", ""),
        "modality_name": series.get("modality", ""),
        "comparative": study.get("comparative", ""),
        "study_id": study.get("study_id", ""),
        "umie_id": v2_record.get("entry_id", ""),
        "mask_path": mask_path,
        "labels": labels,
        "source_labels": list(annotations.get("source_labels", [])),
    }
