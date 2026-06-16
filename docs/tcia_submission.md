# TCIA registration checklist (Task 46)

Per `UMIE notes/UMIE TCIA.md`, UMIE should be listed on **The Cancer Imaging Archive (TCIA)** /
linked from its analysis-results wiki. This is the checklist of metadata to assemble before
submitting; it is an outreach/administrative task with no code dependency.

Reference: <https://wiki.cancerimagingarchive.net/plugins/servlet/mobile?contentId=22515655>

## Required metadata to assemble

- [ ] **Collection / resource name:** UMIE — Unified Medical Imaging dataset.
- [ ] **Short description & purpose:** aggregation of public datasets unified to one convention
      (UMIE ids, folder layout, RadLex labels/masks, JSONL metadata) for pretraining medical imaging models.
- [ ] **Modalities covered:** CT, MRI, X-ray, Mammography (derived from `config/dataset_config.py`).
- [ ] **Source datasets + licences:** the full table from `config/provenance.py` (each source's name,
      URL, licence, redistributability). TCIA-sourced collections (Brain-Tumor-Progression, CT-ORG,
      LIDC-IDRI, CMMD) should be cross-linked to their TCIA pages.
- [ ] **Citation metadata:** `CITATION.cff` (UMIE) + per-source citations (datasheets / provenance).
- [ ] **Access / hosting:** where the processed corpus is hosted and how to obtain it.
- [ ] **Contact / maintainers:** authors from `CITATION.cff`.
- [ ] **Licence for the aggregated release:** must be compatible with the most restrictive source
      licence in any redistributed subset (several sources are non-commercial / non-redistributable).

## Notes

- Do **not** redistribute non-redistributable sources via TCIA; link to the original instead.
- Keep this checklist and `config/provenance.py` in sync — provenance is the machine-readable source
  of truth that the datasheets and HF cards already consume.
