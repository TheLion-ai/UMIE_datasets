# UMIE conventions

These conventions are the backbone of the corpus. They are extended (new optional fields, steps,
datasets) but not changed ad-hoc; the only sanctioned convention updates are recorded under
[Migrations](migrations/phase_to_modality.md).

## UMIE id

Every image gets a unique id:

```
{dataset_uid}_{modality_id}_{study_id}_{img_id}
```

- `dataset_uid` — 2-digit dataset id (e.g. `00` for kits23), from `DatasetArgs.dataset_uid`.
- `modality_id` — numeric key into `DatasetArgs.modalities` (e.g. `0` → `CT`).
- `study_id` — identifies one imaging examination (all images from the same study share it).
- `img_id` — the individual image/slice id.

The id components are **reused** as the keys of the v2 metadata hierarchy (see [schema](schema.md)).

## Folder layout

```
{dataset_uid}_{dataset_name}/
  {modality}/                 # the modality *value*, e.g. CT, Xray, MRI, MG
    Images/                   # one PNG per slice (or .nii.gz in 3D / combined mode)
    Masks/                    # segmentation masks, same basename as their image
    Volumes/                  # (combined output mode only) volumetric .nii.gz, Task 41
  {dataset_uid}_{dataset_name}.jsonl   # one metadata record per image
```

`Images` and `Masks` are fixed folder constants (`src/constants.py`).

## Labels & masks (RadLex)

Target labels and masks are defined once in `config/labels.py` and `config/masks.py` as `Label` /
`Mask` dataclasses keyed by RadLex name + id, with `source_names` mapping each source dataset's terms
onto them. They additionally carry an optional RadLex **hierarchy** (`radlex_parent_id`), a multi-axis
decomposition (anatomy / pathology / modifier) and optional SNOMED/FMA/Uberon cross-codes — see
`config/ontology.py`. Existing label/mask ids and emitted values never change.

## Metadata (JSONL)

One JSONL record per image. The default **v1** record is flat (`umie_path`, `umie_id`,
`modality_name`, `labels`, `source_labels`, `mask_path`, …). An opt-in **v2** record mirrors the DICOM
patient→study→series hierarchy. See the [schema reference](schema.md).
