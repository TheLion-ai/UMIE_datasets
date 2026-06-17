# Multi-axis label mapping (Theme L, Task 37)

This table relates current UMIE labels to the BiomedParse-style three-axis decomposition
(**Anatomy / Pathology / Modifier**) described in `UMIE notes/Imaging datasets annotation/How to
annoatate images.md`. The axes are **additive** — the existing flat `radlex_name` label is
unchanged; the axes are an optional richer view exposed as `Label.anatomy` / `Label.pathology` /
`Label.modifier` (each a dual-coded `config.ontology.OntologyTerm` carrying a RadLex term/id and the
source name it came from).

## Sample dataset: `kits23` (kidney CT, KiTS23)

| UMIE label (`radlex_name`)   | Anatomy (RadLex)   | Pathology (RadLex)  | Modifier (RadLex)   | Source label        |
|------------------------------|--------------------|---------------------|---------------------|---------------------|
| ClearCellAdenocarcinoma      | Kidney (RID205)    | Neoplasm (RID3957)  | Malignant (RID15655)| `clear_cell_rcc`    |
| WilmsTumor                   | Kidney (RID205)    | Neoplasm (RID3957)  | —                   | `wilms`             |
| Angiomyolipoma               | Kidney (RID205)    | Neoplasm (RID3957)  | Benign (RID15654)   | `angiomyolipoma`    |

> Only a representative subset is populated inline in `config/labels.py`; the remaining kidney
> subtypes inherit Anatomy = Kidney and Pathology = Neoplasm and differ only by their RadLex
> hierarchy position (see `radlex_parent_id`) and `modifier`.

## Modifier-only example (grade axis)

| UMIE label      | Modifier (source)             | Notes                          |
|-----------------|-------------------------------|--------------------------------|
| Osteoarthritis  | Kellgren-Lawrence grade       | `grades=4` — 4 severity levels |

## How to query

```python
from config import labels

cc = labels.label_by_name("ClearCellAdenocarcinoma")
cc.anatomy.radlex_name    # "Kidney"
cc.pathology.radlex_name  # "Neoplasm"
cc.modifier.radlex_name   # "Malignant"
[a.radlex_name for a in labels.label_ancestors(cc)]
# ['RenalAdenocarcinoma', 'Adenocarcinoma', 'Malignant', 'Neoplasm']
```

## Secondary ontology cross-codes (Task 39)

Anatomy masks carry optional `snomed_id` / `fma_id` / `uberon_id` alongside the primary RadLex code,
enabling cross-terminology linkage (e.g. future OMOP/LOINC alignment). Sources:

| Mask   | SNOMED CT  | FMA       | Uberon          |
|--------|------------|-----------|-----------------|
| Kidney | 64033007   | FMA:7203  | UBERON:0002113  |
| Lung   | 39607008   | FMA:7195  | UBERON:0002048  |
| Liver  | 10200004   | FMA:7197  | UBERON:0002107  |
| Brain  | 12738006   | FMA:50801 | UBERON:0000955  |

Codes are from the official SNOMED CT International, FMA, and Uberon browsers; they are optional and
have no effect on RadLex-based behavior.
