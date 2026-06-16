# Add a dataset

The recipe for integrating a new source dataset. (Onboarding new datasets is Group 2 of the task
plan and is gated behind finishing the Group 1 reorganization + migration.)

## 1. Register the dataset config

Add a `DatasetArgs` to `config/dataset_config.py`:

```python
my_dataset = DatasetArgs(
    dataset_uid="49",                 # next free 2-digit id
    dataset_name="my_dataset",
    modalities={"0": "CT"},           # numeric key -> modality value (folder name)
    labels={ "source_label": [{labels.Neoplasm.radlex_name: 1}] },  # source -> RadLex
    masks={ "source_mask": MaskColor(source_color=1, target_color=masks.Kidney.id) },
)
```

Map any **new** source terms onto existing RadLex `Label` / `Mask` entries (extend `source_names` in
`config/labels.py` / `config/masks.py`); only add a brand-new `Label`/`Mask` if no RadLex concept fits.

## 2. Record provenance

Add the licence + source attribution to `config/provenance.py` (Task 23). This feeds the datasheet,
the HF card and the citation page.

## 3. Write the pipeline

Add `src/pipelines/my_dataset.py` — a `BasePipeline` subclass with:

- a `steps` tuple (reuse the catalog in [steps.md](steps.md)),
- `pipeline_args = PipelineArgs(...)` wiring the id extractors / selectors,
- `prepare_pipeline()` setting any dataset-specific extractors (e.g.
  `self.ctx.identity.modality_id_extractor = MyModalityExtractor(self.ctx.dataset.modalities)`).

Subclass the base extractors (`img_id` / `study_id` / `modality_id` / `label`) and selectors only
where the default behavior doesn't fit.

## 4. Add a golden test

Add `testing/tests/test_my_dataset.py` mirroring the existing per-dataset tests (run the pipeline on
the dummy input, then `verify_file_tree` / `verify_all_images_identical` / `verify_jsonl_correct`).
Upload the dummy input + expected output to the `umie-tests` S3 bucket.

## 5. Generate the datasheet

```bash
python -m utils.datasheet my_dataset --jsonl data/49_my_dataset/49_my_dataset.jsonl
```

## 6. Validate

```bash
python -m utils.audit_datasets ./data           # duplicates / corrupt / imbalance (Task 32)
```
