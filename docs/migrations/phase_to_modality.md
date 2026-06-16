# Migration: `phase` → `modality` (and opt-in JSONL v2)

This note records the convention migration applied to the already-integrated datasets (Theme P,
Task 48) so downstream consumers can adapt. It is the rollout of the sanctioned `phase` → `modality`
rename (Theme O, Task 47).

## What changed on disk

The migration is deliberately minimal: **only one emitted field changed.**

| Item | Before | After |
|------|--------|-------|
| JSONL record key | `"phase_name"` | `"modality_name"` |
| UMIE id value (`{uid}_{phase_id}_{study}_{img}`) | unchanged | unchanged (the segment is a numeric key) |
| Folder layout (`{uid}_{name}/CT/Images\|Masks/`) | unchanged | unchanged (folder names are modality *values* like `CT`) |
| PNG / NIfTI pixel data & filenames | unchanged | unchanged |

In other words: the id values and the entire on-disk file tree are **byte-identical**; the single
observable change is the JSONL key rename `phase_name → modality_name`.

The configuration/code concept was renamed throughout `src/` and `config/` as well:
`DatasetArgs.phases → modalities`, `BasePhaseIdExtractor → BaseModalityIdExtractor` (module
`phase_id.py → modality_id.py`), the per-pipeline `PhaseIdExtractor/PhaseExtractor` subclasses, and
all `phase_id`/`phase_name` locals. A true contrast *phase* (arterial/venous/delayed) is intentionally
left as a separate, future concept (see `src/base/extractors/modality_id.py` and `Modalities
Ontology.md`); it is not modelled by this rename.

## How consumers adapt

- **JSONL readers:** read `record["modality_name"]` instead of `record["phase_name"]`. No other key
  changed; record order and all other values are identical.
- **Pipeline authors:** pass `modalities={...}` to `DatasetArgs` and set
  `ctx.identity.modality_id_extractor`. There are **no compatibility aliases** — this is a clean
  convention update.

## Golden test data

The expected outputs that the `testing/tests/` golden suite compares against were regenerated to use
`modality_name`:

- **Local** `testing/test_dummy_data/<ds>/expected_output/**/*.jsonl` files: updated in place (key
  rename only; values identical). The local golden suite passes on the new convention.
- **S3 (`umie-tests` bucket):** the same regenerated expected JSONLs must be re-uploaded so CI is green
  on the new convention. **This step needs S3 credentials and is performed by a maintainer** (run the
  pipelines, then push the regenerated `expected_output` JSONLs). The file-tree and image golden checks
  are unaffected by the rename.

## Opt-in additions (not forced by this migration)

The richer **JSONL v2** schema (Theme K) and the **ontology hierarchy / multi-axis** layer (Theme L)
are additive and opt-in — they are *available* to the migrated datasets but are not enabled by default,
so v1 output stays the default. A dataset adopts v2 by setting `PathArgs.schema_version = "2.0"`.

## Post-migration validation

After migrating, run the audit tool over the processed datasets to confirm they are both
convention-correct and quality-clean (Task 32):

```bash
python -m utils.audit_datasets ./data --out audit_findings.md
```

It reports duplicates, corrupt/blank/undersized images and label/modality imbalance per dataset
without modifying any data. Any issues are filed and reviewed — never silently fixed.
