# How to cite UMIE and its sources (Task 46)

UMIE is an *aggregation* of public medical-imaging datasets. Citing it correctly means citing **both**
the UMIE corpus **and** every source dataset whose data you used.

## Cite UMIE

Use the metadata in [`CITATION.cff`](../CITATION.cff) (GitHub renders a "Cite this repository" button
from it):

> Klaudel, Obuchowski, Frąckowski, Komor, Bober, Badyra. *Towards Medical Foundational Model — a
> Unified Dataset for Pretraining Medical Imaging Models.* https://github.com/TheLion-ai/UMIE_datasets

## Cite the source datasets

Each integrated dataset carries its licence, source attribution and (where recorded) a citation in
[`config/provenance.py`](../config/provenance.py), the single source of truth. These are surfaced
automatically in:

- the per-dataset **datasheets** (`python -m utils.datasheet <name>`, Task 43), and
- the **HuggingFace dataset cards** produced by `ExportHuggingFace` (Task 30).

To list every source's licence and citation:

```bash
python -m utils.datasheet            # writes a datasheet per dataset under docs/datasheets/
```

When you publish results, include the citation for **each** source dataset you trained or evaluated on
(see its datasheet's *References* section), and respect each source's licence — several are
non-commercial or non-redistributable (e.g. KiTS23 is CC-BY-NC-SA-4.0, LiTS is CC-BY-NC-ND-4.0).
