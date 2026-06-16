# JSONL metadata schema

Each dataset has a `{uid}_{name}.jsonl` with one record per image. Two schema versions exist; the
version is selected per-run via `PathArgs.schema_version` and defaults to **v1**.

## v1 — flat (default)

Written by `AddUmieIds` + `AddLabels` (+ optional enrichment steps). Fields:

| Field | Description |
| --- | --- |
| `umie_path` | Relative path to the output image (PNG, or `.nii.gz` in 3D mode). |
| `dataset_name` / `dataset_uid` | Dataset identity. |
| `modality_name` | Modality (`CT`, `Xray`, `MRI`, `MG`, …). *(was `phase_name` before the [migration](migrations/phase_to_modality.md))* |
| `comparative` | `PRE` / `POST` / empty, for paired pre/post studies. |
| `study_id` | Study identifier. |
| `umie_id` | The `{uid}_{modality}_{study}_{img}` id. |
| `mask_path` | Relative path to the mask, or empty. |
| `labels` | List of `{radlex_name: grade}` image-level labels. |
| `source_labels` | The original source labels. |

Optional additive fields (opt-in steps): `output_mode` + `volume_metadata` (3D), `license` /
`source_dataset` (Task 23), `split` (Task 21), duplicate/quality flags, extracted DICOM tags.

## v2 — hierarchical (opt-in, `schema_version = "2.0"`)

Built by `ConvertJsonlToV2` (appended automatically when v2 is selected) via `src/metadata_schema.py`,
mirroring `JSONlines.md`. It **reuses** the same id components as the hierarchy keys and can always be
converted back to v1 (`metadata_schema.v2_to_v1`). Top-level structure:

```jsonc
{
  "schema_version": "2.0",
  "entry_id": "00_0_001_5.png",
  "dataset_source_details": { "name": "...", "dataset_uid": "00", "dataset_name": "kits23", ... },
  "patient_level_info": { "patient_id": "..." },                 // de-identified
  "study_level_info":   { "study_id": "001", "study_date_offset_days": null, "comparative": "" },
  "series_level_info": {
    "series_id": "00_0_001_5.png",
    "modality": "CT",
    "imaging_protocol_details": { /* modality-specific, dual _radlex/_source, Task 34 */ },
    "nifti_files":      { "affine_matrix": ..., "original_nifti_dimensions": ..., ... },  // Task 35
    "png_representation": { "type": "native_2d" | "3d_sliced_to_2d", "normalization_applied_to_png": ..., ... }
  },
  "annotations": {
    "image_level_classification": [ { "radlex_label": {"term","id"}, "grade": 1 } ],
    "source_labels": [ ... ],
    "segmentations": [ { "structure_radlex": {"term","id"}, "mask_value_in_nifti": 1, "provenance": "manual" } ]
  }
}
```

### Modality protocol vocabulary (Task 34)

`imaging_protocol_details` fields are controlled per modality (`metadata_schema.MODALITY_PROTOCOL_FIELDS`):
MRI (sequence type, contrast agent/phase, fat-sat, plane), CT (contrast agent/phase, kernel, kVp,
low-dose), X-ray (projection AP/PA/Lateral), MG (CC/MLO), PET (radiotracer, scan type), plus common
slice thickness / pixel spacing. Key terms carry dual `{field}_radlex` `{term,id}` + `{field}_source`,
validated against the controlled vocabulary.

### 3D geometry (Tasks 40–41)

`nifti_files` is the source of truth for 3D geometry (affine, dims, voxel spacing — reused from the
3D `volume_metadata`, not recomputed). When a volume is sliced to PNGs, `png_representation` records
the slicing axis and the actual intensity transform; `src/geometry.py` can reconstruct a volume from
the slices. Combined output mode (`OutputMode.SLICES_2D_AND_VOLUMES_3D`) keeps both the PNG slices and
the `.nii.gz` volume, cross-referenced.
