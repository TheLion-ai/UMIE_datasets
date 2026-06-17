# UMIE datasets documentation

UMIE unifies many public medical-imaging datasets into a single convention so they can be used
together to pretrain medical imaging models. This documentation covers the pipeline architecture, the
UMIE conventions, the step catalog, the add-a-dataset recipe and the JSONL schema reference.

It complements — and does not duplicate — the repository's
[`README.md`](https://github.com/TheLion-ai/UMIE_datasets/blob/main/README.md) (project overview) and
[`get_started.md`](https://github.com/TheLion-ai/UMIE_datasets/blob/main/get_started.md) (environment
setup and running the pipelines).

## Map

- **[Conventions](conventions.md)** — UMIE ids, folder layout, `Images`/`Masks`, RadLex labels/masks.
- **[Step catalog](steps.md)** — every pipeline step and what it does.
- **[Add a dataset](add_a_dataset.md)** — the recipe for integrating a new source dataset.
- **[JSONL schema](schema.md)** — the v1 flat record and the optional v2 hierarchical record.
- **[Citation](citation.md)** — how to cite UMIE and its sources.
- **[Migrations](migrations/phase_to_modality.md)** — sanctioned convention updates.

## Architecture in one paragraph

Each source dataset has a `BasePipeline` subclass (`src/pipelines/`) built from an ordered tuple of
**steps** (`src/steps/`). Steps receive a structured `PipelineContext` (paths, dataset config, id
extractors, selectors, and the opt-in Theme D–N sub-configs) and transform the data in place,
converting source images to PNG (or NIfTI), unifying ids/labels/masks to the UMIE convention, and
writing a JSONL metadata record per image. Extension points — `BaseStep`, the extractors
(`img_id`/`study_id`/`modality_id`/`label`), the selectors, `DatasetArgs` and `PathArgs` — are reused
rather than duplicated.
