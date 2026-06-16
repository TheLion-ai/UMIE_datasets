# Step catalog

Every pipeline is an ordered tuple of steps (`src/steps/`, registered in `src/steps/__init__.py`).
Each is a `BaseStep` receiving the shared `PipelineContext`. The optional Theme D–N steps are
**additive and opt-in** — off / no-op by default, never changing UMIE ids, the folder layout or
existing JSONL fields.

## Core pipeline

| Step | Purpose |
| --- | --- |
| `GetFilePaths` | Collect source file paths. |
| `CreateFileTree` | Create the `{uid}_{name}/{modality}/Images\|Masks/` output tree. |
| `StoreSourcePaths` / `CreateFileToDcmAttributeMapping` | Map outputs back to their source. |
| `AddUmieIds` | Assign UMIE ids, write the JSONL records. |
| `AddLabels` | Fill `labels` / `source_labels` from the per-pipeline label extractor. |
| `ValidateData` | Validate the produced records and tree. |

## Conversion

| Step | Purpose |
| --- | --- |
| `ConvertDcm2Png` / `ConvertNii2Png` / `ConvertJpg2Png` / `ConvertTif2Png` | Source → 2D PNG. |
| `ConvertDcm2Nii` / `ConvertNii2Nii` | Source → 3D `.nii.gz` (VOLUMES_3D mode, Tasks 6–7). |
| `CopyMasks` / `RecolorMasks` / `MasksToBinaryColors` / `CreateBlankMasks` / `CreateMasksFromXml` | Mask handling. |

## Data quality (Theme D)

`DetectDuplicates` (Task 10), `DetectCorruptImages` (Task 11), `CheckMaskQuality` (Task 12),
`ValidateDicomMetadata` (Task 13).

## Preprocessing (Theme E)

`ApplyWindowing` (Task 14), `ApplyClahe` (Task 15), `NormalizeSpacing` (Task 16), `ResizeImages`
(Task 17), `StandardizeBitDepth` (Task 18), `AutocropBorders` (Task 19).

## Metadata & format (Themes F–G)

`ExtractDicomMetadata` (Task 20), `CreateSplits` (Task 21), `AddProvenance` (Task 23),
`ConvertDicomSeg` (Task 24), `ConvertBboxToMask` (Task 25), `MergeMasks` (Task 26).

## Infrastructure, export & schema (Themes H–K, M)

`CreateManifest` (Task 27), `SkipProcessed` (Task 28), `ExportHuggingFace` (Task 30),
`ConvertJsonlToV2` (Task 33 — appended automatically when `schema_version = "2.0"`),
`StoreVolumesAlongside` (Task 41 — appended automatically in combined output mode).

## Standalone utilities (not pipeline steps)

`utils/distribution_report.py` (Task 22), `utils/audit_datasets.py` (Task 32),
`utils/datasheet.py` (Task 43), `utils/subset.py` (Task 45),
`utils/technical_validation.py` (Task 44).
